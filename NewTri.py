import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
import random
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep
from torch.optim import Adam
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
from src.utils import resume_train_state
from src.eval import Trip_M_val as val
from src.optimizer import LinearWarmupCosineAnnealingLR, CosineAnnealingWarmRestarts

from torch_ema import ExponentialMovingAverage

# model
from src.models.rendezvous import Rendezvous
# from src.models.MambaOnly import TriBase
from src.models.MO import TriBase
from src.models.RIT import RiT
# from src.models.Swin import TripletModel
from src.models.DualModel import TripletModel, PublicClassify
# from simclr.modules.loss import NT_Xent
from torchmetrics import Metric
# from pl_bolts.losses.self_supervised import nt_xent_loss
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

def random_crop_from_upsampled(x, scale_factor=1.2):
    _, _, W, H = x.shape
    tensor = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    _, _, new_W, new_H = tensor.shape

    assert new_W >= W and new_H >= H

    start_W = random.randint(0, new_W - W )
    start_H = random.randint(0, new_H - H )

    cropped_tensor = tensor[:, :, start_W:start_W + W, start_H:start_H + H]

    return cropped_tensor

def contrastive_loss(z1, z2, temperature=0.2):
    # z1 = F.normalize(z1, dim=1)
    # z2 = F.normalize(z2, dim=1)
    try:
        batch_size, num_features, channels = z1.size()
        z1 = z1.view(batch_size, -1)
        z2 = z2.view(batch_size, -1)
    except:
        z1 = z1
        z2 = z2
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def get_pre_loss(cls, fm, out_image, mask_feature, mfm, mcls, img, accelerator, step):
    mse_loss = nn.MSELoss()
    ar = 0.1
    ac = 0.1
    bc = 0.005
    ar_loss = mse_loss(out_image, img)
    sr_loss = mse_loss(mask_feature, fm)
    hcl_loss  = contrastive_loss(fm, mfm, temperature=0.2)
    lcl_loss = contrastive_loss(cls, mcls, temperature=0.2)
    
    drc_loss = ar * ar_loss + (1-ar) * sr_loss
    hlcl = ac * hcl_loss + bc * lcl_loss
    
    loss = drc_loss + hlcl
    accelerator.log({
        'Train/Total Pre Loss': float(loss.item()),
        'Train/hcl_loss': float(ac * hcl_loss.item()),
        'Train/ar_loss': float(ar * ar_loss.item()),
        'Train/sr_loss': float((1-ar) *sr_loss.item()),
        'Train/lcl_loss': float(bc * lcl_loss.item())
    }, step=step)
    return loss
 
def get_class_loss(config, output, labels, accelerator, step):
    (y1, y2, y3, y4) = labels
    logit_ivt = output[:, :100]
    logit_i = output[:, 100:106]
    logit_v = output[:, 106:116]
    logit_t = output[:, 116:]
    
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    alpha_instrument, alpha_verb, alpha_target = get_focal_weight_balancing(config)
    
    loss_fn_i = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device))
    loss_fn_v = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device))
    loss_fn_t = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device))
    loss_fn_ivt = nn.BCEWithLogitsLoss()
    focal_loss_i = FocalLoss(alpha=torch.tensor(alpha_instrument).to(accelerator.device), gamma=2.0)
    focal_loss_v = FocalLoss(alpha=torch.tensor(alpha_verb).to(accelerator.device), gamma=2.0)
    focal_loss_t = FocalLoss(alpha=torch.tensor(alpha_target).to(accelerator.device), gamma=2.0)
    
    # class loss             
    loss_i       = loss_fn_i(logit_i, y1.float())
    loss_v       = loss_fn_v(logit_v, y2.float())
    loss_t       = loss_fn_t(logit_t, y3.float())
    loss_ivt     = loss_fn_ivt(logit_ivt, y4.float())  
    focal_loss_i = focal_loss_i(logit_i, y1.float())
    focal_loss_v = focal_loss_v(logit_v, y2.float())
    focal_loss_t = focal_loss_t(logit_t, y3.float())
    focal_loss   = focal_loss_i + focal_loss_v + focal_loss_t
    
    # total loss
    loss = (loss_i) + (loss_v) + (loss_t) + loss_ivt + focal_loss
    # log
    accelerator.log({
        'Train/Total Class Loss': float(loss.item()),
        'Train/loss_i': float(loss_i.item()),
        'Train/loss_v': float(loss_v.item()),
        'Train/loss_t': float(loss_t.item()),
        'Train/focal_loss_i': float(focal_loss_i.item()),
        'Train/focal_loss_v': float(focal_loss_v.item()),
        'Train/focal_loss_t': float(focal_loss_t.item()),
        'Train/loss_ivt': float(loss_ivt.item()),
    }, step=step)
    return loss   

def train_one_epoch(config, model, momentummodel, ema, train_loader, optimizer, scheduler, accelerator, epoch, step):
    model.train()
    momentummodel.eval()
    # with torch.cuda.amp.autocast(enabled=True):
    for batch, (img, (y1, y2, y3, y4)) in enumerate(train_loader):
        cls, fm, out_image, mask_feature = model(img)
        
        mimg = random_crop_from_upsampled(img, scale_factor=1.2)
        
        with torch.no_grad():
            _, mfm, _, _  = momentummodel(mimg)
        
            try:
                mcls = model.model.head(model.model.norm(mfm).mean(dim=1))
            except:
                mcls = model.module.model.head(model.module.model.norm(mfm).mean(dim=1))
        
        pre_loss   = get_pre_loss(cls, fm, out_image, mask_feature, mfm, mcls, img, accelerator, step)
        class_loss = get_class_loss(config, cls, (y1, y2, y3, y4), accelerator, step)
        
        loss = 0.05 * pre_loss +  1* class_loss
        accelerator.backward(loss)
        
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        
        optimizer.step()
        
        model.zero_grad()
        
        ema.update()
        
        step += 1
        accelerator.print(
                f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Training Losses => total:[{loss.item():.4f}] pre: [{pre_loss.item():.4f}]  class: [{class_loss.item():.4f}]', flush=True)
    
    scheduler.step()
    
    # ema cpoy
    ema.store()  
    ema.copy_to(momentummodel.parameters())
    
    if config.trainer.val_training == True:
        metrics, _ = val(config, model, train_loader, activation, step=-1, train=True)
        accelerator.log(metrics, step=epoch)
    
    return step
            
def val_one_epoch(config, model, val_loader, activation, epoch, step):
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
    
    # model
    model         = TripletModel(model_name='swin_base_patch4_window7_224')
    momentummodel = TripletModel(model_name='swin_base_patch4_window7_224')
    
    # load dataset
    train_loader, val_loader, test_loader = give_dataset(config)
    
    # training tools
    optimizer = Adam(
            model.parameters(),
            lr=0.0002,
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
    
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    alpha_instrument, alpha_verb, alpha_target = get_focal_weight_balancing(config)
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    
    # resume
    if config.trainer.resume.train:
        model, optimizer, scheduler, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(model, config.finetune.checkpoint + config.trainer.dataset, optimizer, scheduler, accelerator)
    
    # set in devices
    model, momentummodel, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(model, momentummodel, train_loader, val_loader, optimizer, scheduler)
    
    # ema
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    
    for epoch in range(start_num_epochs, config.trainer.num_epochs):
        train_step = train_one_epoch(config, model, momentummodel,  ema, train_loader, optimizer, scheduler, accelerator, epoch, train_step)
        score, metrics, val_step = val_one_epoch(config, model, val_loader, activation, epoch, val_step)
        
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