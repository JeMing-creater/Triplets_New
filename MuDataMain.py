import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep
from torch.optim import Adam
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from open_clip import create_model_from_pretrained, get_tokenizer
# src
# from src.dataloader_s import give_dataset
from src.dataloader import give_dataset
from src.optimizer import give_scheduler
from torch.utils.data import Dataset, DataLoader
from src.utils import same_seeds, corrupt, _extract_into_tensor, Logger, get_focal_weight_balancing, get_weight_balancing, set_param_in_device, step_params, load_pretrain_model, FocalLoss
from src.utils import resume_train_state_d as resume_train_state
from src.eval import Trip_T_val as val
from src.optimizer import LinearWarmupCosineAnnealingLR, CosineAnnealingWarmRestarts

# model
from src.models.G_F import FussionModel, CholecT45, add_tokens_tokenizer, get_all_list, labels
from src.models.Swin import TripletModel
from loss import *

config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))


def give_training_setting(config, modelG, modelC):
    optimizerG = Adam(
            modelG.parameters(),
            lr=float(config.trainer.Flr[0]),
            weight_decay=0,
            amsgrad=False,
    )
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.6)
    optimizerC = Adam(
            modelC.parameters(),
            lr=float(config.trainer.Flr[1]),
            weight_decay=float(config.trainer.weight_decay),
            amsgrad=False,
    )
    schedulerC = CosineAnnealingWarmRestarts(
            optimizerC,
            T_0=(config.trainer.num_epochs +1),
            T_mult=1,
            eta_min=2e-5,
            last_epoch=-1,
        )
    return optimizerG, schedulerG, optimizerC, schedulerC


def train_one_epoch_fussion(config, modelG, modelC, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, step, best_loss):
    modelC.eval()
    modelG.train()
    all_loss = 0
    all_count = 0
    for batch, (images, texts, (_, _, _, y4)) in enumerate(train_loader):
        # generate output
        output = modelG(images, texts)
        # triple output
        tri_output = modelC(output)
        logit_ivt = tri_output[:, :100]
        # compute loss
        # _, _, rows, columns = images.shape
        # weighttemp = int(np.sqrt(rows * columns))
        # Loss_LpLssim, _, _  = loss_functions['loss1'](image_in=images, image_out=output, weight=weighttemp)
        Loss_L1, _, _       = loss_functions['loss2'](image_vis=images, generate_img=output)
        Loss_Classify       = loss_functions['loss3'](logit_ivt, y4.float())
        Total_loss          = Loss_L1  + Loss_Classify
        all_loss += Total_loss
        all_count += 1
        
        # back ward loss
        accelerator.backward(Total_loss)
        
        # for name, param in modelG.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        # optimizer.step
        optimizer.step()
        # modelG.zero_grad
        modelG.zero_grad()
        # log 
        accelerator.log({
            'Train Setp2/Total Loss': float(Total_loss.item()),
            # 'Train/text_loss': float(text_loss.item()), 
            'Train Setp2/Loss L1': float(Loss_L1.item()),
            # 'Train Setp2/Loss LpLssim': float((0.5 * Loss_LpLssim).item()),
            'Train Setp2/Loss Classify': float((0.5 * Loss_Classify).item()),
        }, step=step)
        step += 1
        accelerator.print(
                f'Epoch [{epoch+1}/{config.trainer.fuss_epochs}][{batch + 1}/{len(train_loader)}] Step2 Training Loss: Total Loss:[{Total_loss.item():.4f}] Loss L1: [{Loss_L1.item():.4f}] Loss Classify: [{(Loss_Classify).item():.4f}]', flush=True)
    # scheduler.step()
    scheduler.step()
    all_loss = all_loss / all_count
    
    # save
    if all_loss < best_loss:
        best_loss = all_loss
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step2/best/new/")
        torch.save(modelG.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step2/best/new/model.pth")
    
    return step, best_loss


def train_one_epoch(config, modelG, modelC, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, step):
    # train
    modelG.eval()
    modelC.train()
    for batch, (img, text, (y1, y2, y3, y4)) in enumerate(train_loader):
        # output 4 result
        if config.trainer.dataset == 'T50':
            b, m, c, h, w = img.size()
            img = img.view(-1, c, h, w)
        
        g_output  = modelG(img, text)
        output    = modelC(g_output)
        logit_ivt = output[:, :100]
        logit_i   = output[:, 100:106]
        logit_v   = output[:, 106:116]
        logit_t   = output[:, 116:]
        
        if config.trainer.dataset == 'T50':
            logit_i   = logit_i.view(b, m, -1)[:, -1, :]
            logit_v   = logit_v.view(b, m, -1)[:, -1, :]
            logit_t   = logit_t.view(b, m, -1)[:, -1, :]
            logit_ivt = logit_ivt.view(b, m, -1)[:, -1, :]
        
        # class loss             
        loss_i       = loss_functions['loss_fn_i'](logit_i, y1.float())
        loss_v       = loss_functions['loss_fn_v'](logit_v, y2.float())
        loss_t       = loss_functions['loss_fn_t'](logit_t, y3.float())
        loss_ivt     = loss_functions['loss_fn_ivt'](logit_ivt, y4.float())  
        focal_loss_i = loss_functions['focal_loss_i'](logit_i, y1.float())
        focal_loss_v = loss_functions['focal_loss_v'](logit_v, y2.float())
        focal_loss_t = loss_functions['focal_loss_t'](logit_t, y3.float())
        focal_loss   = focal_loss_i + focal_loss_v + focal_loss_t
        
        # total loss
        loss = (loss_i) + (loss_v) + (loss_t) + loss_ivt + focal_loss
        
        # lose backward
        accelerator.backward(loss)
        
        # optimizer.step
        optimizer.step()
        
        modelC.zero_grad()
        
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
        }, step=step)
        step += 1
        accelerator.print(
                f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Training Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}]  i: [{loss_i.item():.4f}, {focal_loss_i.item():.4f}] v: [{loss_v.item():.4f}, {focal_loss_v.item():.4f}] t: [{loss_t.item():.4f}, {focal_loss_t.item():.4f}]', flush=True)
    # learning rate schedule update
    scheduler.step()
    
    if config.trainer.val_training == True:
        metrics, _ = val(config, modelC, train_loader, activation, step=-1, train=True)
        accelerator.log(metrics, step=epoch)
    
    return step

def val_one_epoch(config, model, val_loader, loss_functions, activation, epoch, step):
    metrics, step = val(config, model, val_loader, activation, step=step, train=False)
    i_score = metrics['Val/I']
    t_score = metrics['Val/T']
    v_score = metrics['Val/V']
    ivm_score = metrics['Val/IVM']
    itm_score = metrics['Val/ITM']
    ivt_score = metrics['Val/IVT']
    accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Val Metrics => ivt: [{ivt_score}] i: [{i_score}] v: [{v_score}] t: [{t_score}] iv: [{ivm_score}] it: [{itm_score}] ', flush=True)    
    accelerator.log(metrics, step=epoch)
    return ivt_score, metrics, step

  
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
    
    # models
    modelG = FussionModel()
    modelC = TripletModel(model_name='swin_base_patch4_window7_224')
    
    # dataloader
    add_list = get_all_list(labels)
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    add_tokens_tokenizer(tokenizer, add_list)
    dataset = CholecT45( 
                tokenizer,        
                dataset_dir='/root/.cache/huggingface/forget/datasets/CholecT45/', 
                dataset_variant='cholect45-crossval',
                test_fold=1,
                augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                )
    
    train_dataset, val_dataset, test_dataset = dataset.build()
    batch_size = config.dataset.T45.batch_size
    num_workers = config.dataset.T45.num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=3*batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=False)
    
    # training tools
    optimizerG, schedulerG, optimizerC, schedulerC = give_training_setting(config, modelG, modelC)
    
    
    # Step1
    # TODO: train a class model, use Trimain.py, and get the best path
    base_path = '/workspace/jeming/Trip/model_store/Swin_OT45/best/new'
    modelC = load_pretrain_model(base_path + "/model.pth", modelC, accelerator)
    
    # set in devices
    modelC, modelG, train_loader, val_loader, optimizerG, schedulerG, optimizerC, schedulerC = accelerator.prepare(modelC, modelG, train_loader, val_loader, optimizerG, schedulerG, optimizerC, schedulerC)
    
    # Step2
    # training the FussionModel to create mutli model data.
    # loss_functions = {
    #     # loss1: 预测图 与 原图 LpLssim 差距
    #     'loss1': LpLssimLossweight().to(accelerator.device),
    #     # loss1: 预测图 与 原图 l1_loss 差距
    #     'loss2': Fusionloss().to(accelerator.device),
    #     # loss3: 生成图 --> modelC --> 分类损失
    #     'loss3': nn.BCEWithLogitsLoss()
    # }
    # best_loss = 10000.0
    # start_num_epochs = 0
    # train_step = 0
    # # training
    # for epoch in range(start_num_epochs, config.trainer.fuss_epochs):
    #     train_step, best_loss = train_one_epoch_fussion(config, modelG, modelC, train_loader, loss_functions, optimizerG, schedulerG, accelerator, epoch, train_step, best_loss)
    #     # check point
    #     accelerator.print('Checkout....')
    #     accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step2/checkpoint/")
    #     torch.save({'epoch': epoch, 'best_loss': best_loss,  'train_step': train_step},
    #                 f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step2/checkpoint/epoch.pth.tar')
    #     accelerator.print('Checkout Over!')
    
    
    # Step3
    # loading FussionModel and generate image for classify.
    # TODO: need to train modelG first.
    accelerator.print('Step3!')
    base_path = '/workspace/jeming/Trip/model_store/MuDataT45/Step2/best/new'
    modelG = load_pretrain_model(base_path + "/model.pth", modelG, accelerator)
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    
    # loss functions
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    alpha_instrument, alpha_verb, alpha_target = get_focal_weight_balancing(config)
    loss_functions = {
        'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
        'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
        'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
        'focal_loss_i': FocalLoss(alpha=torch.tensor(alpha_instrument).to(accelerator.device), gamma=2.0),
        'focal_loss_v': FocalLoss(alpha=torch.tensor(alpha_verb).to(accelerator.device), gamma=2.0),
        'focal_loss_t': FocalLoss(alpha=torch.tensor(alpha_target).to(accelerator.device), gamma=2.0),
        'loss_fn_ivt': nn.BCEWithLogitsLoss(),
    }
    
    for epoch in range(start_num_epochs, config.trainer.num_epochs):
        train_step = train_one_epoch(config, modelG, modelC, train_loader, loss_functions, optimizerC, schedulerC, accelerator, epoch, train_step)
        score, metrics, val_step = val_one_epoch(config, modelC, val_loader, loss_functions, activation, epoch, val_step)
        
        # save best model
        if best_score.item() < score:
            best_score = score
            best_metrics = metrics
            # two types of modeling saving
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step3/best/new/")
            torch.save(modelC.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step3/best/new/model.pth")
            torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step3/best/epoch.pth.tar')
            
        # print best score
        accelerator.print(f'Now best APscore: {best_score}', flush=True)
        
        # checkout
        accelerator.print('Checkout....')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step3/checkpoint")
        torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/Step3/checkpoint/epoch.pth.tar')
        accelerator.print('Checkout Over!')