import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import sys
import pytz
import yaml
import monai
import torch
import ivtmetrics
from torch import nn
from typing import Dict
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep
from torch.optim import Adam
import torchvision.transforms as T

# src
from src.dataloader_s import give_dataset
from src.optimizer import give_scheduler
from torch.utils.data import Dataset, DataLoader
from src.utils import same_seeds, Logger, get_weight_balancing, set_param_in_device, step_params, load_pretrain_model, FocalLoss
from src.utils import resume_train_state_d as resume_train_state
from src.eval import Trip_val as val
from src.optimizer import LinearWarmupCosineAnnealingLR, CosineAnnealingWarmRestarts

# model
from src.models.rendezvous import Rendezvous
from src.models.RIT import RiT
from src.models.Swin import TripletModel
from diffusers import DDPMScheduler, UNet2DModel
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

def give_train_setting(config, model, step=1):
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    alpha_instrument = torch.tensor([0.2, 0.6, 0.3, 0.8, 0.7, 0.65])
    alpha_verb = torch.tensor([0.3, 0.1, 0.2, 0.5, 0.6, 0.7, 0.6, 0.8, 0.9, 0.4])
    alpha_target = torch.tensor([0.1, 0.55, 0.4, 0.5, 0.4, 0.85, 0.75, 0.7, 0.3, 0.8, 0.35, 0.65, 0.7, 0.45, 0.4]) 
    if step == 1:
        optimizer = Adam(
            model.parameters(),
            lr=float(config.trainer.Tlr[0]),
            weight_decay=float(config.trainer.weight_decay),
            amsgrad=False,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=(config.trainer.num_epochs +1),
            T_mult=1,
            eta_min=2e-5,
            last_epoch=-1,
        )
        loss_functions = {
            'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
            'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
            'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
            'focal_loss_i': FocalLoss(alpha=alpha_instrument, gamma=2.0),
            'focal_loss_v': FocalLoss(alpha=alpha_verb, gamma=2.0),
            'focal_loss_t': FocalLoss(alpha=alpha_target, gamma=2.0),
            'loss_fn_ivt': nn.BCEWithLogitsLoss(),
        }
        return optimizer, scheduler, loss_functions
    elif step == 2:
        optimizer = Adam(
            model.parameters(),
            lr=float(config.trainer.Tlr[1]),
            weight_decay=float(config.trainer.weight_decay),
            amsgrad=False,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=(config.trainer.step2_epochs +1),
            T_mult=1,
            eta_min=2e-5,
            last_epoch=-1,
        )
        loss_functions = {
            'MSE_loss' : nn.MSELoss(),
            'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
            'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
            'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
            'loss_fn_ivt': nn.BCEWithLogitsLoss(),
        }
        return optimizer, scheduler, loss_functions

def Step1(config, netC, activation, train_loader, val_loader, accelerator):
    optimizer, scheduler, loss_functions = give_train_setting(config, netC, step=1) 
    netC, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(netC, train_loader, val_loader, optimizer, scheduler)
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    # resume
    if config.trainer.resume.train:
        netC, optimizer, scheduler, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(netC, config.finetune.checkpoint + config.trainer.dataset, optimizer, scheduler, accelerator, step=1)
    
    # training
    if config.trainer.is_train:
        for epoch in range(start_num_epochs, config.trainer.num_epochs):
            # train
            netC.train()
            for batch, (img, (y1, y2, y3, y4)) in enumerate(train_loader):
                # output 4 result
                if config.trainer.dataset == 'T50':
                    b, m, c, h, w = img.size()
                    img = img.view(-1, c, h, w)
                
                output = netC(img)
                triple = output[:, :100]
                tool = output[:, 100:106]
                verb = output[:, 106:116]
                target = output[:, 116:]
                loss_i       = loss_functions['loss_fn_i'](tool, y1.float())
                loss_v       = loss_functions['loss_fn_v'](verb, y2.float())
                loss_t       = loss_functions['loss_fn_t'](target, y3.float())
                loss_ivt     = loss_functions['loss_fn_ivt'](triple, y4.float())  
                focal_loss_i = loss_functions['focal_loss_i'](tool, y1.float())
                focal_loss_v = loss_functions['focal_loss_v'](verb, y2.float())
                focal_loss_t = loss_functions['focal_loss_t'](target, y3.float())
                focal_loss   = focal_loss_i + focal_loss_v + focal_loss_t
                loss         = (loss_i) + (loss_v) + (loss_t) + loss_ivt + focal_loss
                
                # lose backward
                accelerator.backward(loss)
                optimizer.step()
                netC.zero_grad()
                # log
                accelerator.log({
                    'Train/Total Loss': float(loss.item()),
                    'Train/loss_i': float(loss_i.item()),
                    'Train/loss_v': float(loss_v.item()),
                    'Train/loss_t': float(loss_t.item()),
                    'Train/focal_loss_i': float(focal_loss_i.item()),
                    'Train/focal_loss_v': float(focal_loss_v.item()),
                    'Train/focal_loss_t': float(focal_loss_t.item()),
                    'Train/loss_ivt': float(loss_ivt.item()),
                }, step=train_step)
                train_step += 1
                accelerator.print(
                    f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Step1 Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f}, {focal_loss_i.item():.4f}] v: [{loss_v.item():.4f}, {focal_loss_v.item():.4f}] t: [{loss_t.item():.4f}, {focal_loss_t.item():.4f}]', flush=True)
            # scheduler
            scheduler.step()
            if config.trainer.val_training == True:
                metrics, _ = val(config, netC, train_loader, activation, step=-1, train=True, train_step='1')
                accelerator.log(metrics, step=epoch)
            
            # val
            metrics, val_step = val(config, netC, val_loader, activation, step=val_step, train=False, train_step='1')
            i_score = metrics['Val1/I']
            t_score = metrics['Val1/T']
            v_score = metrics['Val1/V']
            ivm_score = metrics['Val1/IVM']
            itm_score = metrics['Val1/ITM']
            ivt_score = metrics['Val1/IVT']
            accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Step 1 Val Metrics => ivt: [{ivt_score}] i: [{i_score}] v: [{v_score}] t: [{t_score}] iv: [{ivm_score}] it: [{itm_score}] ', flush=True)    
            accelerator.log(metrics, step=epoch)
            
            # save best model
            if best_score.item() < ivt_score:
                best_score = ivt_score
                best_metrics = metrics
                # two types of modeling saving
                accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/step1/")
                torch.save(netC.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/step1/best_model.pth")
            
            # print best score
            accelerator.print(f'Now best APscore: {best_score}', flush=True)
            
            # checkout
            accelerator.print('Checkout....')
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint1")
            torch.save(netC.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint1/checkpoint_model.pth")
            torch.save(optimizer.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint1/checkpoint_optimizer.pth")
            torch.save(scheduler.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint1/checkpoint_scheduler.pth")
            torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                        f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint1/epoch.pth.tar')
            accelerator.print('Checkout Over!')
            
    accelerator.print(f"Step1 ivt score: {best_score}")
    accelerator.print(f"other metrics : {best_metrics}")
 
                
def Step2(config, netC, netG, activation, train_loader, val_loader, accelerator):
    optimizer, scheduler, loss_functions = give_train_setting(config, netG, step=2) 
    netC, netG, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(netC, netG, train_loader, val_loader, optimizer, scheduler)
    
    

        
if __name__ == '__main__':
    same_seeds(50)
    # log
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config), flush=True)
    
    # activation
    activation = nn.Sigmoid()
    
    # load dataset
    train_loader, val_loader, test_loader = give_dataset(config)
    
    # model
    netC = TripletModel(model_name='swin_base_patch4_window7_224')
    netG = UNet2DModel(
            sample_size=224,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(224, 448, 512),  # Roughly matching our basic unet example
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",   # a regular ResNet upsampling block
            ),
        )
    
    # setp1
    Step1(config, netC, activation, train_loader, val_loader, accelerator)
    Step2(config, netC, netG, activation, train_loader, val_loader, accelerator)