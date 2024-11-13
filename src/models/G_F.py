
import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, TFGPT2LMHeadModel
from typing import Tuple
import numpy as np
import torch.nn as nn
from PIL import Image
import timm
import math
import numbers
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from flask import Flask, jsonify
# from net import TransformerBlock as Restormer
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms

labels = [
    'Grasper dissects cystic plate.',
    'Grasper dissects gallbladder.',
    'Grasper dissects omentum.',
    'Grasper grasps cystic artery.',
    'Grasper grasps cystic duct.',
    'Grasper grasps cystic pedicle.',
    'Grasper grasps cystic plate.',
    'Grasper grasps gallbladder.',
    'Grasper grasps gut.',
    'Grasper grasps liver.',
    'Grasper grasps omentum.',
    'Grasper grasps peritoneum.',
    'Grasper grasps specimen bag.',
    'Grasper packs gallbladder.',
    'Grasper retracts cystic duct.',
    'Grasper retracts cystic pedicle.',
    'Grasper retracts cystic plate.',
    'Grasper retracts gallbladder.',
    'Grasper retracts gut.',
    'Grasper retracts liver.',
    'Grasper retracts omentum.',
    'Grasper retracts peritoneum.',
    'Bipolar coagulates abdominal wall cavity.',
    'Bipolar coagulates blood vessel.',
    'Bipolar coagulates cystic artery.',
    'Bipolar coagulates cystic duct.',
    'Bipolar coagulates cystic pedicle.',
    'Bipolar coagulates cystic plate.',
    'Bipolar coagulates gallbladder.',
    'Bipolar coagulates liver.',
    'Bipolar coagulates omentum.',
    'Bipolar coagulates peritoneum.',
    'Bipolar dissects adhesion.',
    'Bipolar dissects cystic artery.',
    'Bipolar dissects cystic duct.',
    'Bipolar dissects cystic plate.',
    'Bipolar dissects gallbladder.',
    'Bipolar dissects omentum.',
    'Bipolar grasps cystic plate.',
    'Bipolar grasps liver.',
    'Bipolar grasps specimen bag.',
    'Bipolar retracts cystic duct.',
    'Bipolar retracts cystic pedicle.',
    'Bipolar retracts gallbladder.',
    'Bipolar retracts liver.',
    'Bipolar retracts omentum.',
    'Hook coagulates blood vessel.',
    'Hook coagulates cystic artery.',
    'Hook coagulates cystic duct.',
    'Hook coagulates cystic pedicle.',
    'Hook coagulates cystic plate.',
    'Hook coagulates gallbladder.',
    'Hook coagulates liver.',
    'Hook coagulates omentum.',
    'Hook cuts blood vessel.',
    'Hook cuts peritoneum.',
    'Hook dissects blood vessel.',
    'Hook dissects cystic artery.',
    'Hook dissects cystic duct.',
    'Hook dissects cystic plate.',
    'Hook dissects gallbladder.',
    'Hook dissects omentum.',
    'Hook dissects peritoneum.',
    'Hook retracts gallbladder.',
    'Hook retracts liver.',
    'Scissors coagulate omentum.',
    'Scissors cut adhesion.',
    'Scissors cut blood vessel.',
    'Scissors cut cystic artery.',
    'Scissors cut cystic duct.',
    'Scissors cut cystic plate.',
    'Scissors cut liver.',
    'Scissors cut omentum.',
    'Scissors cut peritoneum.',
    'Scissors dissect cystic plate.',
    'Scissors dissect gallbladder.',
    'Scissors dissect omentum.',
    'Clipper clips blood vessel.',
    'Clipper clips cystic artery.',
    'Clipper clips cystic duct.',
    'Clipper clips cystic pedicle.',
    'Clipper clips cystic plate.',
    'Irrigator aspirates fluid.',
    'Irrigator dissects cystic duct.',
    'Irrigator dissects cystic pedicle.',
    'Irrigator dissects cystic plate.',
    'Irrigator dissects gallbladder.',
    'Irrigator dissects omentum.',
    'Irrigator irrigates abdominal wall cavity.',
    'Irrigator irrigates cystic pedicle.',
    'Irrigator irrigates liver.',
    'Irrigator retracts gallbladder.',
    'Irrigator retracts liver.',
    'Irrigator retracts omentum.',
    'Only grasper.',
    'Only bipolar.',
    'Only hook.',
    'Only scissors.',
    'Only clipper.',
    'Only irrigator.'
]

class CholecT45():
    def __init__(self,
                 tokenizer,
                 dataset_dir, context_length=50, image_size=[224,224],
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90']):
        self.image_size = image_size
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
            'contrast': transforms.RandomAutocontrast(p=0.5),
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
            dataset = T45(tokenizer=self.tokenizer,context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
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
            dataset = T45(tokenizer=self.tokenizer,context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
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
            dataset = T45( tokenizer=self.tokenizer,context_length = self.context_length,
                          img_dir=os.path.join(self.dataset_dir, 'data', video),
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
    def __init__(self, tokenizer, context_length, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels = [
    'Grasper dissects cystic plate.',
    'Grasper dissects gallbladder.',
    'Grasper dissects omentum.',
    'Grasper grasps cystic artery.',
    'Grasper grasps cystic duct.',
    'Grasper grasps cystic pedicle.',
    'Grasper grasps cystic plate.',
    'Grasper grasps gallbladder.',
    'Grasper grasps gut.',
    'Grasper grasps liver.',
    'Grasper grasps omentum.',
    'Grasper grasps peritoneum.',
    'Grasper grasps specimen bag.',
    'Grasper packs gallbladder.',
    'Grasper retracts cystic duct.',
    'Grasper retracts cystic pedicle.',
    'Grasper retracts cystic plate.',
    'Grasper retracts gallbladder.',
    'Grasper retracts gut.',
    'Grasper retracts liver.',
    'Grasper retracts omentum.',
    'Grasper retracts peritoneum.',
    'Bipolar coagulates abdominal wall cavity.',
    'Bipolar coagulates blood vessel.',
    'Bipolar coagulates cystic artery.',
    'Bipolar coagulates cystic duct.',
    'Bipolar coagulates cystic pedicle.',
    'Bipolar coagulates cystic plate.',
    'Bipolar coagulates gallbladder.',
    'Bipolar coagulates liver.',
    'Bipolar coagulates omentum.',
    'Bipolar coagulates peritoneum.',
    'Bipolar dissects adhesion.',
    'Bipolar dissects cystic artery.',
    'Bipolar dissects cystic duct.',
    'Bipolar dissects cystic plate.',
    'Bipolar dissects gallbladder.',
    'Bipolar dissects omentum.',
    'Bipolar grasps cystic plate.',
    'Bipolar grasps liver.',
    'Bipolar grasps specimen bag.',
    'Bipolar retracts cystic duct.',
    'Bipolar retracts cystic pedicle.',
    'Bipolar retracts gallbladder.',
    'Bipolar retracts liver.',
    'Bipolar retracts omentum.',
    'Hook coagulates blood vessel.',
    'Hook coagulates cystic artery.',
    'Hook coagulates cystic duct.',
    'Hook coagulates cystic pedicle.',
    'Hook coagulates cystic plate.',
    'Hook coagulates gallbladder.',
    'Hook coagulates liver.',
    'Hook coagulates omentum.',
    'Hook cuts blood vessel.',
    'Hook cuts peritoneum.',
    'Hook dissects blood vessel.',
    'Hook dissects cystic artery.',
    'Hook dissects cystic duct.',
    'Hook dissects cystic plate.',
    'Hook dissects gallbladder.',
    'Hook dissects omentum.',
    'Hook dissects peritoneum.',
    'Hook retracts gallbladder.',
    'Hook retracts liver.',
    'Scissors coagulate omentum.',
    'Scissors cut adhesion.',
    'Scissors cut blood vessel.',
    'Scissors cut cystic artery.',
    'Scissors cut cystic duct.',
    'Scissors cut cystic plate.',
    'Scissors cut liver.',
    'Scissors cut omentum.',
    'Scissors cut peritoneum.',
    'Scissors dissect cystic plate.',
    'Scissors dissect gallbladder.',
    'Scissors dissect omentum.',
    'Clipper clips blood vessel.',
    'Clipper clips cystic artery.',
    'Clipper clips cystic duct.',
    'Clipper clips cystic pedicle.',
    'Clipper clips cystic plate.',
    'Irrigator aspirates fluid.',
    'Irrigator dissects cystic duct.',
    'Irrigator dissects cystic pedicle.',
    'Irrigator dissects cystic plate.',
    'Irrigator dissects gallbladder.',
    'Irrigator dissects omentum.',
    'Irrigator irrigates abdominal wall cavity.',
    'Irrigator irrigates cystic pedicle.',
    'Irrigator irrigates liver.',
    'Irrigator retracts gallbladder.',
    'Irrigator retracts liver.',
    'Irrigator retracts omentum.',
    'Only grasper.',
    'Only bipolar.',
    'Only hook.',
    'Only scissors.',
    'Only clipper.',
    'Only irrigator.'
]

    def __len__(self):
        return len(self.triplet_labels)

    def add_text(self, text):
        num = text.size(0)
        text = text.sum(dim=0, keepdim=True) / num
        return text.squeeze(0)
    
    def __getitem__(self, index):
        # template = 'this is a photo of '
        triplet_label = self.triplet_labels[index, 1:]
        text_labels = [item for label, item in zip(triplet_label, self.labels) if label == 1]
        if text_labels == []:
            text = 'A scenario with no tools or corresponding actions.'
        else:
            # for label in text_labels:
            add_text = ''
            for label in labels:
                add_text = add_text + ' ' + label
            add_text = add_text 
            text = f'A laparoscopic cholecystectomy scenario with [{add_text}]'
        
        texts = self.tokenizer(text, context_length=self.context_length)
        # texts = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=100)
        # input_ids = texts['input_ids']
        # attention_mask = texts['attention_mask']
        # input_ids = input_ids.squeeze()
        # attention_mask = attention_mask.squeeze()
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

        texts = self.add_text(texts)
        
        return image, texts, (tool_label, verb_label, target_label, triplet_label)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FeedForwards(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(FeedForwards, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Linear(dim, hidden_features)
        self.dwconv = nn.Linear(hidden_features, hidden_features)
        self.project_out = nn.Linear(hidden_features//2, dim)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
   
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, ffn_expansion_factor=4):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        # 这里我们使用线性层来生成Q, K, V，但在实际应用中，Q, K, V可以是不同的线性层
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feedforward = FeedForwards(dim = embed_dim, ffn_expansion_factor = ffn_expansion_factor)
        
    def forward(self, inputs):
        image_encoding, text_encoding = inputs
        # 将文本编码结果作为Q，图像编码结果作为K和V
        query = text_encoding
        key = image_encoding
        value = image_encoding
        
        attn_output, _ = self.multihead_attn(query, key, value)
        
        out = self.feedforward(attn_output)
        
        return (out, query)

class ImageDecoder(nn.Module):
    def __init__(self, embed_dim=512, ffn_expansion_factor=4, num_heads=8, out_channels = 3, image_size=(224, 224)):
        super().__init__()  
        self.image_size = image_size
        self.Ts1 = TransformerBlock(dim = math.ceil(math.pow(embed_dim, 1/3)), num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')
        self.ups1 = nn.ConvTranspose2d(in_channels=math.ceil(math.pow(embed_dim, 1/3)), out_channels=math.ceil(math.pow(embed_dim, 1/3))*2, kernel_size=4, stride=4, padding=0)
        self.Ts2 = TransformerBlock(dim = math.ceil(math.pow(embed_dim, 1/3))*2, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')
        self.ups2 = nn.ConvTranspose2d(in_channels=math.ceil(math.pow(embed_dim, 1/3))*2, out_channels=out_channels, kernel_size=4, stride=4, padding=0)
        self.Ts3 = TransformerBlock(dim = out_channels, num_heads=out_channels, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], math.ceil(math.pow(inputs.size()[1], 1/3)), math.ceil(math.pow(inputs.size()[1], 1/3)), math.ceil(math.pow(inputs.size()[1], 1/3)))
        inputs = self.Ts1(inputs)
        inputs = self.ups1(inputs)
        inputs = self.Ts2(inputs)
        inputs = self.ups2(inputs)
        inputs = self.Ts3(inputs)
        inputs = F.interpolate(inputs, size=(self.image_size[0], self.image_size[0]), mode='bilinear', align_corners=False)
        inputs = self.sigmoid(inputs)
        return inputs

class FussionModel(nn.Module):
    def __init__(self, block_num = 4, out_channels = 3, num_heads=4, ffn_expansion_factor=4, image_size=(224, 224)):
        super().__init__()
        self.model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        block = []
        for _ in range(block_num):
            block.append(CrossAttention(embed_dim=512, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor))
        self.coss_att = nn.Sequential(*block)
        self.image_decoder = ImageDecoder(embed_dim=512, ffn_expansion_factor=ffn_expansion_factor, num_heads=num_heads, out_channels = out_channels, image_size=image_size)
        # un update param
        self.model.logit_scale.requires_grad = False
            
    def forward(self, image, texts):
        image_features, text_features, _ = self.model(image, texts.to(torch.long))
        out, _ = self.coss_att((image_features, text_features))
        out = self.image_decoder(out)
        return image + out

def add_tokens_tokenizer(tokenizer, all_list):
    add = []
    for word in all_list:
        if word in tokenizer.tokenizer.vocab:
            pass
        else:
            print(f"'{word}' is not in the BERT vocabulary.")
            add.append(word)
    num_added_toks = tokenizer.tokenizer.add_tokens(add)
    print('Now we have added', num_added_toks, 'tokens')
    return tokenizer

def get_all_list(labels):
    add_list = []
    for label in labels:
        for word in label.split():
            if word.replace('.', '') not in add_list:
                add_list.append(word.replace('.', ''))
    return add_list

if __name__ == '__main__':
    device = 'cuda:0'
    
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    # input_text = f"Please help me generate a single complete caption of a laparoscopic cholecystectomy scenario with multiple action subjects: [{labels[0]} {labels[1]}]"
    # input_ids = tokenizer.encode(input_text, return_tensors='tf')
    # output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(output_text)
    # # 加载处理器和模型
    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, prefetch_factor=3*20, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=20, shuffle=False, prefetch_factor=3*20, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False)
    
    model = FussionModel().to(device)
    
    for batch, (images, texts, (y1, y2, y3, y4)) in enumerate(train_dataloader):
        images = images.to(device)
        texts  = texts.to(device)
        out = model(images, texts)
        
        
        
        
        
        
        
        
        
        
        
        
        
        # print(input_ids.shape)
        # print(attention_mask.shape)
        
    
    # # 加载处理器和模型
    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    # text = f'A laparoscopic cholecystectomy scenario with [{labels[0]}, {labels[1]}]'
    
    # max_length = 100
    
    # inputs = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    
    # inputs = inputs.to(device)
    

    