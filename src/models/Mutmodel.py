import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
from typing import Tuple
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from open_clip import create_model_from_pretrained, get_tokenizer

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
            text_labels.append('no tools or corresponding actions.')
        # if text_labels != []:
        #     t = ''
        #     for l in text_labels:
        #         if t == '':
        #             t = l.replace('.', '')
        #         else:
        #             t = t + ' and ' + l.replace('.', '')
        #     text = template + t + '.'
        # else:
        #     text = 'no tools or corresponding actions.'
        texts = self.tokenizer([l for l in text_labels], context_length=self.context_length)
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
            # print(f"Success: {img_path}")
        except Exception as e:
            # print(f"Fail: {img_path} - {e}")
            # Handle the failure by creating a blank image or skipping
            image = Image.new('RGB', (256, 448), color=(255, 255, 255))  # Use a blank image as a placeholder

        texts = self.add_text(texts)
        
        return image, (tool_label, verb_label, target_label, triplet_label, texts)
    

class TripletModel(nn.Module):
    def __init__(self, model_name, class_num=131, context_length=50, pretrained=True):
        super().__init__()
        self.output_feature = {} 
        """
        Models class to return swin transformer models
        """
        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # print(self.model)
        # Get the number features in final embedding
        n_features = self.model.head.in_features

        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, class_num)
        
        self.model.layers.register_forward_hook(self.get_activation('image feature'))
        # Update the classification layer with our custom target size
        # self.linear_layer = nn.Linear(49 * 1024, 50)
        self.text_head = nn.Linear(49 * 1024, context_length)

    # def add_text(self, text):
    #     num = text.size(1)
    #     text = text.sum(dim=1, keepdim=True) / num
    #     return text
    def get_activation(self, layer_name):
        def hook(module, input: Tuple[torch.Tensor], output:torch.Tensor):
            self.output_feature[layer_name] = output
        return hook
    
    def forward(self, x):
        x = self.model(x)
        # text_march = self.text_head(x)
        image_march = self.output_feature['image feature'].view(x.size()[0], -1)
        # text_march = self.text_head(image_march)
        return x, image_march


if __name__ == '__main__':
    device = 'cuda:0'
    context_length = 50
    image = torch.randn(2, 3, 224, 224).to(device)
    
    text = [labels[0], labels[1]]
    
    model = TripletModel(model_name='swin_base_patch4_window7_224').to(device)
    
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # texts = tokenizer([l for l in text], context_length=context_length).to(device)
    # texts = texts.unsqueeze(0).expand(image.size()[0], -1, -1)
    
    dataset = CholecT45( 
                tokenizer,        
                dataset_dir='/root/.cache/huggingface/forget/datasets/CholecT45/', 
                dataset_variant='cholect45-crossval',
                test_fold=1,
                augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                )
    
    train_dataset, val_dataset, test_dataset = dataset.build()
    
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, prefetch_factor=3*20, num_workers=1, pin_memory=True, persistent_workers=True, drop_last=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=20, shuffle=False, prefetch_factor=3*20, num_workers=1, pin_memory=True, persistent_workers=True, drop_last=False)
    
    for batch, (images, (y1, y2, y3, y4, texts)) in enumerate(train_dataloader):
        # print(texts.shape)
        output = model(images.to(device))
        text = output[1]
        # print(images.shape)
    
    
    # output = model(image, texts)
    
    # triple = output[:, :100]
    # tool = output[:, 100:106]
    # verb = output[:, 106:116]
    # target = output[:, 116:]
    # print(output.shape)