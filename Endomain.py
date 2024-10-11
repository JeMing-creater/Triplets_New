import os
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
from transformers import AutoTokenizer
# src
from src.dataloader import give_dataset
# from src.txtdataloader import give_dataset
from src.optimizer import give_scheduler, LinearWarmupCosineAnnealingLR
from src.utils import same_seeds, Logger, get_weight_balancing, set_param_in_device, step_params, resume_train_state, load_pretrain_model, add_tokens_tokenizer
from src.eval import val, PA_val
# model
from src.models.rendezvous import Rendezvous
from src.models.RIT import RiT
from src.models.NewPA import PA
from src.models.EndoForm import EndoForm

# config setting
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

   
def train_one_epoch(config, model, activation, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, step):
    # train
    model.train()
    for batch, (img, (y1, y2, y3, y4)) in enumerate(train_loader):
        # output 4 result
        if config.trainer.dataset == 'T50':
            b, m, c, h, w = img.size()
            img = img.view(-1, c, h, w)
        
        tool, verb, target, triplet = model(img)
        logit_i  = tool
        logit_v  = verb
        logit_t  = target
        logit_ivt   = triplet
        
        if config.trainer.dataset == 'T50':
            logit_i   = logit_i.view(b, m, -1)[:, -1, :]
            logit_v   = logit_v.view(b, m, -1)[:, -1, :]
            logit_t   = logit_t.view(b, m, -1)[:, -1, :]
            logit_ivt = logit_ivt.view(b, m, -1)[:, -1, :]
                        
        loss_i      = loss_functions['loss_fn_i'](logit_i, y1.float())
        loss_v      = loss_functions['loss_fn_v'](logit_v, y2.float())
        loss_t      = loss_functions['loss_fn_t'](logit_t, y3.float())
        loss_ivt    = loss_functions['loss_fn_ivt'](logit_ivt, y4.float())  
        loss        = (loss_i) + (loss_v) + (loss_t) + loss_ivt 
        
        # lose backward
        accelerator.backward(loss)
        
        # optimizer.step
        optimizer.step()
        optimizer.zero_grad()
        
        # model.zero_grad()
        
        # log
        accelerator.log({
            'Train/Total Loss': float(loss.item()),
            'Train/loss_i': float(loss_i.item()),
            'Train/loss_v': float(loss_v.item()),
            'Train/loss_t': float(loss_t.item()),
            'Train/loss_ivt': float(loss_ivt.item()),
        }, step=step)
        step += 1
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f}] v: [{loss_v.item():.4f}] t: [{loss_t.item():.4f}]', flush=True)
        # break
    # learning rate schedule update
    scheduler.step(epoch)
    accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Epoch Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f}] v: [{loss_v.item():.4f}] t: [{loss_t.item():.4f}]', flush=True)    

    if config.trainer.val_training == True:
        metrics, _ = val(config, model, train_loader, activation, step=-1, train=True)
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
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config), flush=True)
    
    # # tokenizer and add word
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
    # target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'othertarget']       
    # verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'others']      
        
    # all_list = instrument_list + target_list + verb_list
    # tokenizer = add_tokens_tokenizer(tokenizer, all_list)
    
    # # load dataset
    train_loader, val_loader, test_loader = give_dataset(config)
    
    # load model
    model = EndoForm(in_channels=3)
    
    # optimizer
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr[0], betas=(0.9, 0.95))
    
    # scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    # activation
    activation = nn.Sigmoid()
    
    # loss
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    loss_functions = {
        'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
        'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
        'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
        'loss_fn_ivt': nn.BCEWithLogitsLoss(),
    }
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    
    # resume
    if config.trainer.resume.train:
        model, optimizer, scheduler, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(model, config.finetune.checkpoint + config.trainer.dataset, optimizer, scheduler, accelerator)
    if config.trainer.resume.test:
        model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/new/pytorch_model.bin", model, accelerator)
    
    # set in device
    model, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, val_loader, optimizer, scheduler)
    
    # training
    if config.trainer.is_train == True:
        if config.trainer.is_train:
            for epoch in range(start_num_epochs, config.trainer.num_epochs):
                # train
                train_step = train_one_epoch(config, model, activation, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, train_step)
                score, metrics, val_step = val_one_epoch(config, model, val_loader, loss_functions, activation, epoch, val_step)
                
                # save best model
                if best_score.item() < score:
                    best_score = score
                    best_metrics = metrics
                    # two types of modeling saving
                    accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/new/")
                    torch.save(model.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/new/model.pth")
                    torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                            f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/best/epoch.pth.tar')
                    
                # print best score
                accelerator.print(f'Now best APscore: {best_score}', flush=True)
                
                # checkout
                accelerator.print('Checkout....')
                accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint")
                torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                            f'{os.getcwd()}/model_store/{config.finetune.checkpoint + config.trainer.dataset}/checkpoint/epoch.pth.tar')
                accelerator.print('Checkout Over!')
    
    # val
    if config.trainer.is_train != True:
        best_score, best_metrics, val_step = val_one_epoch(config, model, val_loader, loss_functions, activation, epoch, val_step)
     
    accelerator.print(f"dice ivt score: {best_score}")
    accelerator.print(f"other metrics : {best_metrics}")
    sys.exit(1) 
            
    