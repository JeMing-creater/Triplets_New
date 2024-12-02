import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
import pytz
import yaml
import monai
import torch
from PIL import Image
import ivtmetrics
from torch import nn
from typing import Dict
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as T
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep
from torch.optim import Adam
import numpy as np
import json
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
from src.utils import same_seeds, corrupt, _extract_into_tensor, freeze, unfreeze, Logger, get_focal_weight_balancing, get_weight_balancing, set_param_in_device, step_params, load_pretrain_model, FocalLoss
from src.utils import resume_train_state_d as resume_train_state
from src.eval import Trip_C_val as val
from src.optimizer import LinearWarmupCosineAnnealingLR, CosineAnnealingWarmRestarts

# model
# from src.models.LViTPrediction import CholecT45
from transformers import CLIPProcessor, CLIPModel
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, AutoTokenizer
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel, MedCLIPProcessor

config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

class CholecT45():
    def __init__(self,
                 tokenizer,processor,
                 dataset_dir, context_length=200, image_size=[224,224],
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90']):
        self.image_size = image_size
        self.processor = processor
        self.context_length = context_length
        # self.preprocess = preprocess
        self.tokenizer = tokenizer
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
        # op_test = [transforms.Resize((self.image_size[0], self.image_size[1])), transforms.ToTensor(), normalize, ]
        # op_train = [transforms.Resize((self.image_size[0], self.image_size[1]))] + self.augmentation_list + [transforms.Resize((self.image_size[0], self.image_size[1])),
        #                                                                        transforms.ToTensor(), normalize, ]
        op_test = []
        op_train = self.augmentation_list 
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T45(tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
                          words_file=os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
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
            dataset = T45(tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
                          words_file=os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
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
            dataset = T45( tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
                          words_file=os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform)
            iterable_dataset.append(dataset)
        self.test_dataset = ConcatDataset(iterable_dataset)

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)

class T45(Dataset):
    def __init__(self, tokenizer, processor, context_length, img_dir, words_file, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.context_length = context_length
        self.words_labels = self.load_text_by_index(words_file)
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        
    def load_text_by_index(self, file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 分割每一行的索引和文本，假设它们之间用逗号分隔
                parts = line.strip().split(',', 1)  # 分割一次，确保只分割出索引和文本
                if len(parts) == 2 and parts[0].isdigit():  # 确保第一部分是数字
                    index = int(parts[0])  # 将索引转换为整数
                    text = parts[1]  # 获取文本
                    data[index] = text
        return data
    
    def __len__(self):
        return len(self.triplet_labels)

    def add_text(self, text):
        num = text.size(0)
        text = text.sum(dim=0, keepdim=True) / num
        return text.squeeze(0)
    
    def __getitem__(self, index):
        # template = 'this is a photo of '
        triplet_label = self.triplet_labels[index, 1:]
        
        text = self.words_labels.get(index, None)
        
        # text_labels = [item for label, item in zip(triplet_label, self.labels) if label == 1]
        # if text_labels == []:
        #     text_labels.append('A scenario with no tools or corresponding actions.')
        # else:
        #     txt = ''
        #     for t in text_labels:
        #         txt = txt + t
        #     text_labels = []
        #     text_labels.append(txt)
        
        # texts = self.tokenizer(text, context_length=self.context_length)
        
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        
        
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)

        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                triplet_label = self.target_transform(triplet_label)
        except Exception as e:
            image = Image.new('RGB', (256, 448), color=(255, 255, 255))  # Use a blank image as a placeholder

        # texts = self.add_text(texts)
        
        image = self.processor(image)
        text = self.tokenizer(text, context_length = self.context_length).squeeze(0)
        # inputs = self.processor(
        #         text=text, 
        #         images=image, 
        #         return_tensors="pt",
        #         truncation=True, 
        #         padding="max_length"
        #         )
        # image = inputs['pixel_values'].squeeze(0)
        # text = inputs['input_ids'].squeeze(0)
        
        return image, text, (tool_label, verb_label, target_label, triplet_label)

class CholecT50():
    def __init__(self, 
                dataset_dir, tokenizer, processor, context_length,
                image_size=(224,224),
                dataset_variant="cholect50-crossval",
                test_fold=1,
                augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                normalize=True,
                m=3):
        self.processor = processor
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.image_size = image_size
        self.normalize   = normalize
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits (recommended)",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
            "cholect50-subset": "specially created for EDU4SDS summer school"
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant, "is not a valid dataset variant")
        video_split  = self.split_selector(case=dataset_variant)
        train_videos = sum([v for k,v in video_split.items() if k!=test_fold], []) if 'crossval' in dataset_variant else video_split['train']
        test_videos  = sum([v for k,v in video_split.items() if k==test_fold], []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos   = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos   = video_split['val']
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records   = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records  = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90,expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.m = m
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()
        self.target_transform = self.to_binary

        self.build_train_dataset(trainform)
        self.build_val_dataset(testform)
        self.build_test_dataset(testform)
    
    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val'  : [8, 12, 29, 50, 78],
                'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val':   [5, 18, 36, 65, 74],
                'test':  [92, 96, 103, 110, 111]
            },
            'cholect45-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
            },
            'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # op_test   = [transforms.Resize((self.image_size[0], self.image_size[1])), transforms.ToTensor(), ]
        # op_train  = [transforms.Resize((self.image_size[0], self.image_size[1]))] + self.augmentation_list + [transforms.Resize((self.image_size[0], self.image_size[1])), transforms.ToTensor()]
        # if self.normalize:
        #     op_test.append(normalize)
        #     op_train.append(normalize)
        # testform  = transforms.Compose(op_test)
        # trainform = transforms.Compose(op_train)
        op_test = []
        op_train = self.augmentation_list 
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform
    
    def to_binary(self, label_list):
        outputs = []
        for label in label_list:
            label = torch.tensor(label).bool().int()
            outputs.append(label)
        return outputs

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T50(tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          words_file = os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          words_file = os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)
        # self.val_dataset = iterable_dataset

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(tokenizer=self.tokenizer, processor=self.processor, context_length = self.context_length,
                          img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)), 
                          words_file = os.path.join(self.dataset_dir, 'words', '{}.txt'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.test_dataset = ConcatDataset(iterable_dataset)
        # self.test_dataset = iterable_dataset
        
    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)
    
class T50(Dataset):
    def __init__(self, tokenizer, processor, context_length, img_dir, words_file, label_file, transform=None, target_transform=None, m=3):
        label_data = json.load(open(label_file, "rb"))
        self.label_data = label_data["annotations"]
        self.tokenizer = tokenizer
        self.processor = processor
        self.words_labels = self.load_text_by_index(words_file)
        self.context_length = context_length
        self.frames = list(self.label_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.m = m
        
    def __len__(self):
        return len(self.frames)

    def load_text_by_index(self, file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 分割每一行的索引和文本，假设它们之间用逗号分隔
                parts = line.strip().split(',', 1)  # 分割一次，确保只分割出索引和文本
                if len(parts) == 2 and parts[0].isdigit():  # 确保第一部分是数字
                    index = int(parts[0])  # 将索引转换为整数
                    text = parts[1]  # 获取文本
                    data[index] = text
        return data
    
    def get_binary_labels(self, labels):
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([100])
        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] += 1
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            target = label[8:14]  
            if target[0] != -1.0:   
                target_label[target[0]] += 1       
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        return (tool_label, verb_label, target_label, triplet_label, phase_label)
    
    def __getitem__(self, index):
        current_frame_id = self.frames[index]
        
        # indices = list(map(lambda x: x+int(current_frame_id) if x+int(current_frame_id) > 0 else 0, np.arange(-self.m, 1)))
        # print("indices >> ", indices)

        labels = self.label_data[current_frame_id]
        # images = [Image.open(os.path.join(self.img_dir, f"{int(i):06}.png")) for i in indices]
        
        images = Image.open(os.path.join(self.img_dir, f"{int(current_frame_id):06}.png"))
        labels = self.get_binary_labels(labels)
        text = self.words_labels.get(int(current_frame_id), None)
        
        
        if self.transform:
            # images = list(map(lambda x: self.processor(self.transform(x)), images))
            images = self.processor(self.transform(images))
        if self.target_transform:
            labels = self.target_transform(labels)

        text = self.tokenizer(text, context_length = self.context_length).squeeze(0)
        # return np.stack(images, axis=0), (labels[0], labels[1], labels[2], labels[3])
        return images, text, (labels[0], labels[1], labels[2], labels[3])


# class CLIPclass(nn.Module):
#     def __init__(self, class_num=131):
#         super().__init__()
#         # self.model, _ = clip.load("ViT-B/32", device=accelerator.device)
#         # self.model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         # self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
#         self.model.from_pretrained()
#         # self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
#         # self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
        
#         # self.align = nn.Linear(1024, 768)
#         # out_channel = self.model.visual_projection.out_features    
#         self.fc = nn.Linear(512, 512)
#         self.head = nn.Linear(512, class_num)
    
#     def forward(self, image_input, text_inputs = None):
#         image_features = self.model.encode_image(image_input)
        
#         if self.training == True:
#             text_features = self.model.encode_text(text_inputs)
#             logits_per_image = self.model.compute_logits(image_features, text_features)
#             logits_per_text = logits_per_image.t()
#             class_result = self.fc(image_features.float())
#             class_result = torch.relu(class_result)
#             class_result = self.head(class_result)      
#             return class_result, logits_per_image, logits_per_text
#         else:
#             class_result = self.fc(image_features.float())
#             class_result = torch.relu(class_result)
#             class_result = self.head(class_result)
#             return class_result 
        
class CLIPclass(nn.Module):
    def __init__(self, class_num=131):
        super().__init__()
        # self.model, _ = clip.load("ViT-B/32", device=accelerator.device)
        self.model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        self.fc = nn.Linear(self.model.context_length, self.model.context_length)
        self.head = nn.Linear(self.model.context_length, class_num)
    
    def forward(self, image_input, text_inputs = None):
        image_features = self.model.encode_image(image_input)
        if self.training == True:
            text_features = self.model.encode_text(text_inputs)
            # image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_feature = text_features / text_features.norm(dim=-1, keepdim=True)
            image_feature = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]
            text_feature = F.normalize(text_features, p=2, dim=1)
            
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_feature @ text_feature.t()
            logits_per_text = logit_scale * text_feature @ image_feature.t()
            
            class_result = self.fc(image_features.float())
            class_result = torch.relu(class_result)
            class_result = self.head(class_result)
            
            return class_result, logits_per_image, logits_per_text
        else:
            class_result = self.fc(image_features.float())
            class_result = torch.relu(class_result)
            class_result = self.head(class_result)
            return class_result    


def train_one_epoch(config, model, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, step):
    # train
    model.train()
    # try:
    #     model.vision_model.post_layernorm.requires_grad_(False)
    # except:
    #     model.module.vision_model.post_layernorm.requires_grad_(False)
    for batch, (images, texts, (y1, y2, y3, y4)) in enumerate(train_loader):
        
        output, logits_per_image, logits_per_text = model(images, texts)
        
        logit_ivt = output[:, :100]
        logit_i   = output[:, 100:106]
        logit_v   = output[:, 106:116]
        logit_t   = output[:, 116:]
        
        # target       = torch.arange(images.size(0), device=accelerator.device)
        # loss_image   = loss_functions['loss_ce'](logits_per_image, target)
        # loss_text    = loss_functions['loss_ce'](logits_per_text, target)
        loss_image = nn.functional.cross_entropy(logits_per_image, torch.arange(len(logits_per_image), device=logits_per_image.device))
        loss_text = nn.functional.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device))
        loss_ce      = (loss_image + loss_text) / 2 
            
        loss_i       = loss_functions['loss_fn_i'](logit_i, y1.float())
        loss_v       = loss_functions['loss_fn_v'](logit_v, y2.float())
        loss_t       = loss_functions['loss_fn_t'](logit_t, y3.float())
        loss_ivt     = loss_functions['loss_fn_ivt'](logit_ivt, y4.float())  
        focal_loss_i = loss_functions['focal_loss_i'](logit_i, y1.float())
        focal_loss_v = loss_functions['focal_loss_v'](logit_v, y2.float())
        focal_loss_t = loss_functions['focal_loss_t'](logit_t, y3.float())
        focal_loss   = focal_loss_i + focal_loss_v + focal_loss_t
    
        # total loss
        # if epoch  < config.trainer.pre_epochs:
        #     accelerator.print('pre training')
        #     loss =  loss_ce + 0.0 * ((loss_i) + (loss_v) + (loss_t) + loss_ivt + focal_loss)
        # else:
        #     accelerator.print('class training')
        loss =  config.trainer.word_radio * loss_ce + ((loss_i) + (loss_v) + (loss_t) + loss_ivt + focal_loss)
            
        accelerator.log({
            'Train/Total Class Loss': float(loss.item()),
            'Train/loss_ce': float(loss_ce.item()),
            'Train/loss_i': float(loss_i.item()),
            'Train/loss_v': float(loss_v.item()),
            'Train/loss_t': float(loss_t.item()),
            'Train/focal_loss_i': float(focal_loss_i.item()),
            'Train/focal_loss_v': float(focal_loss_v.item()),
            'Train/focal_loss_t': float(focal_loss_t.item()),
            'Train/loss_ivt': float(loss_ivt.item()),
        }, step=step)
        accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Best [{best_score}] Losses => total:[{loss.item():.4f}] ce: [{loss_ce.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f},{focal_loss_i.item():.4f}] v: [{loss_v.item():.4f},{focal_loss_v.item():.4f}] t: [{loss_t.item():.4f},{focal_loss_t.item():.4f}]', flush=True)    

        
        # lose backward
        accelerator.backward(loss)
        
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        # optimizer.step
        optimizer.step()
        
        model.zero_grad()
    
        step += 1
    # learning rate schedule update
    scheduler.step()
    
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
    # log
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config), flush=True)
    
    # activation
    activation = nn.Sigmoid()
    
    # model
    model = CLIPclass()
    # dataset
    _, processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    dataset = CholecT45( 
                tokenizer, processor, context_length=150,       
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
    
    # setting tools
    # params_to_update = [param for name, param in model.named_parameters() if 'text_model.model.pooler' not in name and param.requires_grad]

    optimizer = Adam(model.parameters(), lr=0.0001,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    # optimizer = Adam(params_to_update, lr=0.0001,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    # optimizer = Adam(params_to_update, lr=0.0001,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # loss
    tool_weight, verb_weight, target_weight = get_weight_balancing(config)
    alpha_instrument, alpha_verb, alpha_target = get_focal_weight_balancing(config)
    loss_functions = {
        # 'loss_ce': nn.CosineEmbeddingLoss(),
        'loss_ce': nn.CrossEntropyLoss(),
        'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
        'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
        'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
        'focal_loss_i': FocalLoss(alpha=torch.tensor(alpha_instrument).to(accelerator.device), gamma=2.0),
        'focal_loss_v': FocalLoss(alpha=torch.tensor(alpha_verb).to(accelerator.device), gamma=2.0),
        'focal_loss_t': FocalLoss(alpha=torch.tensor(alpha_target).to(accelerator.device), gamma=2.0),
        'loss_fn_ivt': nn.BCEWithLogitsLoss()
    }
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    best_metrics = {}
    
    if config.trainer.resume.train:
        model, optimizer, scheduler, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(model, config.finetune.checkpoint + config.trainer.dataset, optimizer, scheduler, accelerator)
        
    model, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, val_loader, optimizer, scheduler)

    
    for epoch in range(start_num_epochs, config.trainer.num_epochs):
        # train
        train_step = train_one_epoch(config, model, train_loader, loss_functions, optimizer, scheduler, accelerator, epoch, train_step)
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
    
        
    