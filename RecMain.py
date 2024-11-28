import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import os
from typing import Tuple
import random
import math
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
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



class CholecT45():
    def __init__(self,
                 dataset_dir, image_size=[224,224],
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90']):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int,
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        self.image_size = image_size
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant,
                                                                          "is not a valid dataset variant")
        video_split = self.split_selector(case=dataset_variant)
        train_videos = sum([v for k, v in video_split.items() if k != test_fold],
                           []) if 'crossval' in dataset_variant else video_split['train']
        test_videos = sum([v for k, v in video_split.items() if k == test_fold],
                          []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos = video_split['val']
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        # 对比增强，不能翻转
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'ccontrast': transforms.RandomAutocontrast(p=0.5),
        }
        
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()
        self.build_train_dataset(trainform)
        self.build_val_dataset(trainform)
        self.build_test_dataset(testform)

    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35,
                          48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val': [8, 12, 29, 50, 78],
                'test': [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25,
                          49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val': [5, 18, 36, 65, 74],
                'test': [92, 96, 103, 110, 111]
            },
            'cholect45-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, ],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, ],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, ],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, ],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, ],
            },
            'cholect50-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        op_test = [transforms.Resize((self.image_size[0], self.image_size[1])), transforms.ToTensor(), normalize, ]
        op_train = [transforms.Resize((self.image_size[0], self.image_size[1]))] + self.augmentation_list + [transforms.Resize((self.image_size[0], self.image_size[1])),
                                                                               transforms.ToTensor(), normalize, ]
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T45(img_dir=os.path.join(self.dataset_dir, 'data', video), image_size=self.image_size, double_t=True,
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform)
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T45(img_dir=os.path.join(self.dataset_dir, 'data', video),image_size=self.image_size,
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform)
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T45(img_dir=os.path.join(self.dataset_dir, 'data', video),image_size=self.image_size,
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform)
            iterable_dataset.append(dataset)
        self.test_dataset = iterable_dataset

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)


class T45(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, image_size=[224,224], transform=None, target_transform=None, double_t = False):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.double_t = double_t
        self.other_transform = transforms.Compose([transforms.Resize((image_size[0], image_size[1])), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)

        try:
            image = Image.open(img_path)
            if self.transform:
                img = self.transform(image)
            if self.double_t == True:
                img2 = self.other_transform(image)
            if self.target_transform:
                triplet_label = self.target_transform(triplet_label)
            # print(f"Success: {img_path}")
        except Exception as e:
            # print(f"Fail: {img_path} - {e}")
            # Handle the failure by creating a blank image or skipping
            img = Image.new('RGB', (256, 448), color=(255, 255, 255))  # Use a blank image as a placeholder
        if self.double_t == True:
            return img2, img, (tool_label, verb_label, target_label, triplet_label)
        else:
            return img, (tool_label, verb_label, target_label, triplet_label)

class UpsampleDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(UpsampleDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # 假设输入的128维向量已经被展平为(batch_size, 128, 1, 1)
        self.fc = nn.Linear(input_size, 128 * 14 * 14)  # 将输入展平并映射到一个中间特征图
        
        # 将特征图上采样到14x14
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 将特征图上采样到28x28
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 将特征图上采样到56x56
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 将特征图上采样到112x112
        self.conv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 将特征图上采样到224x224
        # self.conv5 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # 将输入向量映射到一个中间特征图
        x = self.fc(x)
        x = x.view(-1, 128, 14, 14)  # 重塑为(batch_size, 128, 14, 14)
        
        # 通过转置卷积层进行上采样
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        # x = torch.tanh(self.conv5(x))  # 使用tanh激活函数以确保输出在[-1, 1]范围内
        
        return x

class TripletModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', class_num=131, dim=128, pretrained=True):
        super().__init__()

        self.output_feature = {} 

        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # print(self.model)
        self.model.layers.register_forward_hook(self.get_activation('encoder feature'))
        # Get the number features in final embedding
        n_features = self.model.head.in_features

        self.mlp = nn.Sequential(nn.Linear(n_features, n_features), nn.BatchNorm1d(n_features), nn.ReLU(), nn.Linear(n_features, dim))
        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, class_num)
        
        self.decoder = UpsampleDecoder(input_size=128, output_size=(3, 224, 224))
        
    def get_activation(self, layer_name):
        def hook(module, input: Tuple[torch.Tensor], output:torch.Tensor):
            self.output_feature[layer_name] = output
        return hook
    
    
    def forward(self, x):
        # first
        _ = self.model(x)
        mid_feat = self.output_feature['encoder feature']
        mid_feat = self.model.norm(mid_feat).mean(dim=1)
        
        feat = self.mlp(mid_feat)
        feat = nn.functional.normalize(feat, dim=1)
        
        out_image = self.decoder(feat)
        
        cls = self.model.head(mid_feat)

        return cls, feat, mid_feat, out_image

def get_pre_loss(cls, mcls, feat, mid_feat, out_image, mfeat, mmid_feat, img, accelerator, step):
    mse_ct = nn.MSELoss()
    # kd_ct = torch.nn.KLDivLoss(reduction='none')
    # cls_ct = nn.BCEWithLogitsLoss()
    # loss compute
    ar_loss = mse_ct(out_image, img)
    kd_loss = contrastive_loss(feat, mfeat)
    cls_loss = mse_ct(cls, mcls)
    hcl_loss = contrastive_loss(mid_feat, mmid_feat)
    ar = 0.1
    kd = 0.1
    hcl = 0.1
    cs = 0.005
    
    loss = ar * ar_loss + kd * kd_loss + cs * cls_loss + hcl * hcl_loss
    
    accelerator.log({
        'Train/Total Pre Loss': float(loss.item()),
        'Train/ar_loss': float(ar *  ar_loss.item()),
        'Train/kd_loss': float(kd *  kd_loss.item()),
        'Train/cls_loss': float(cs * cls_loss.item()),
        'Train/hcl_loss': float(hcl * hcl_loss.item())
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
    for batch, (img, mimg, (y1, y2, y3, y4)) in enumerate(train_loader):
        cls, feat, mid_feat, out_image = model(img)
        
        with torch.no_grad():
            _, mfeat, mmid_feat, _  = momentummodel(mimg)
        
            try:
                mcls = model.model.head(mmid_feat)
            except:
                mcls = model.module.model.head(mmid_feat)
        
        pre_loss   = get_pre_loss(cls, mcls, feat, mid_feat, out_image, mfeat, mmid_feat, img, accelerator, step)
        class_loss = get_class_loss(config, cls, (y1, y2, y3, y4), accelerator, step)
        
        loss = config.trainer.pre_radio * pre_loss + 1 * class_loss
        accelerator.backward(loss)
        
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        
        optimizer.step()
        
        model.zero_grad()
        
        ema.update()
        
        step += 1
        accelerator.print(
                f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Best [{best_score}] Training Losses => total:[{loss.item():.4f}] pre: [{pre_loss.item():.4f}]  class: [{class_loss.item():.4f}]', flush=True)
    
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
    
    batch_size = config.dataset.T45.batch_size
    dataset = CholecT45( 
            dataset_dir='/root/.cache/huggingface/forget/datasets/CholecT45/', 
            dataset_variant='cholect45-crossval',
            test_fold=1,
            # augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
            augmentation_list=['contrast','ccontrast'],
            )
    train_dataset, val_dataset, test_dataset = dataset.build()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    
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
        train_step = train_one_epoch(config, model, momentummodel, ema, train_loader, optimizer, scheduler, accelerator, epoch, train_step)
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
    
        