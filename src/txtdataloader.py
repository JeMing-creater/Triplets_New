import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer, BertModel, MobileBertTokenizer, MobileBertModel,BertForMaskedLM


class CholecT50():
    def __init__(self,
                 dataset_dir,
                 tokenizer, 
                 dataset_variant="cholect45-crossval",
                 test_fold=1, text_path='text-data',
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
        self.tokenizer = tokenizer
        self.text_path = text_path
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
        self.instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
        self.target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'othertarget']       
        self.verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'others']      
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
        op_test = [transforms.Resize((256, 448)), transforms.ToTensor(), normalize, ]
        op_train = [transforms.Resize((256, 448))] + self.augmentation_list + [transforms.Resize((256, 448)),
                                                                               transforms.ToTensor(), normalize, ]
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          apply_mask_func=self.apply_mask,  # 增加mask函数
                          get_sentence_func=self.get_random_sentence)  # 增加获取句子的函数
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)


    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          apply_mask_func=self.apply_mask,  # 增加mask函数
                          get_sentence_func=self.get_random_sentence)  # 增加获取句子的函数
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          apply_mask_func=self.apply_mask,  # 增加mask函数
                          get_sentence_func=self.get_random_sentence)  # 增加获取句子的函数
            iterable_dataset.append(dataset)
        self.test_dataset = iterable_dataset

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)

    #随机获取句子
    def get_random_sentence(self, category, label):
        """
        - category (str): 类别，'instrument', 'target', 或 'verb'。
        - label (str): 标签， 'grasper', 'hook', 'action' 等。
        """
        text_folder = os.path.join(self.text_path, category)
        text_file = os.path.join(text_folder, f"{label}.txt")

        if not os.path.exists(text_file):
            raise FileNotFoundError(f"File for {label} under {category} not found.")

        # 打开文本文件并读取前 20 句
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # TODO: 换成 20
            if len(lines) < 10:
                raise ValueError(f"{text_file} does not have 20 sentences.")

            # 从前 20 句中随机选择一句
            random_sentence = random.choice(lines[:10])

        return random_sentence.strip()

    # Mask操作
    def apply_mask(self, instrument, target, verb, max_len=100):
        input_text = '[CLS] ' + instrument + ' [SEP] ' + '[CLS] ' + target + ' [SEP] ' + '[CLS] ' + verb + ' [SEP]'
        input_text = input_text.split()
        masked_index = []
        cls_num = 0

        # 遍历单词并进行mask
        for word_num in range(len(input_text)):
            if cls_num == 1 and input_text[word_num] == instrument:
                input_text[word_num] = '[MASK]'
                masked_index.append(word_num)

            if cls_num == 2 and input_text[word_num] == target:
                input_text[word_num] = '[MASK]'
                masked_index.append(word_num)

            if cls_num == 3 and input_text[word_num] == verb:
                input_text[word_num] = '[MASK]'
                masked_index.append(word_num)

            if input_text[word_num] == '[CLS]':
                cls_num += 1

        # 将句子转为id   
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_text)

        # 使用 max_len 参数进行填充和截断
        if len(indexed_tokens) < max_len:
            # 如果长度不足，使用 0 进行填充
            indexed_tokens += [0] * (max_len - len(indexed_tokens))
        else:
            # 如果长度超出，进行截断
            indexed_tokens = indexed_tokens[:max_len]

        # 将句子转为tensor
        input_tensor = torch.tensor([indexed_tokens])

        # 将句子转为tensor
       # indexed_tokens = tokenizer.convert_tokens_to_ids(input_text)
        #input_tensor = torch.tensor([indexed_tokens])

        return input_tensor

class T50(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None, apply_mask_func=None, get_sentence_func=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.apply_mask_func = apply_mask_func  # 保存mask函数
        self.get_sentence_func = get_sentence_func  # 保存获取句子的函数
        # TODO：确定元素是否按顺序排列
        self.instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
        self.target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'othertarget']       
        self.verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'others']      
      
        
    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)

        #读取图像
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

        # 获取对应的文本
        # TODO: 看情况，如果文本标题是数字，就不用self.instrument_list等，但如果文本标题是文本，就确认一下这些个列表的元素顺序是否对应具体分类名字。
        instrument_sentence = self.get_sentence_func('instrument', np.argmax(tool_label))
        target_sentence = self.get_sentence_func('target', np.argmax(target_label))
        verb_sentence = self.get_sentence_func('verb', np.argmax(verb_label))

        # 对文本进行mask并转换为tensor
        txt_tensor = self.apply_mask_func(instrument_sentence, target_sentence, verb_sentence)

        return image, txt_tensor, (tool_label, verb_label, target_label, triplet_label)


def give_dataset(config, tokenizer):
    dataset = CholecT50( 
            dataset_dir=config.data_dir, 
            dataset_variant=config.dataset_variant,
            test_fold=config.kfold,
            text_path = config.text_path,
            augmentation_list=config.data_augmentations,
            tokenizer = tokenizer
            )
    train_dataset, val_dataset, test_dataset = dataset.build()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, prefetch_factor=3*config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers, drop_last=config.drop_last)
    val_dataloader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, prefetch_factor=3*config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers, drop_last=config.drop_last)
    
    # test data set is built per video, so load differently
    test_dataloaders = []
    for video_dataset in test_dataset:
        test_dataloader = DataLoader(video_dataset, batch_size=config.batch_size, shuffle=False, prefetch_factor=3*config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers, drop_last=config.drop_last)
        test_dataloaders.append(test_dataloader)
    
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    # config setting
    config = EasyDict(yaml.load(open('/workspace/Jeming/TriGit/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    batch_size = 2
    data_dir = '/root/.cache/huggingface/forget/datasets/CholecT45/'
    dataset_variant = 'cholect45-crossval'
    kfold = 1
    data_augmentations = ['original', 'vflip', 'hflip', 'contrast', 'rot90']
    
    # 初始化BERT tokenizer, 注意tokenizer要作为超参传入函数，不能全局定义
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
    target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'null_target']       
    verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'null_verb']      
        
    all_list = instrument_list + target_list + verb_list

    # TODO: 要将一些特殊词加到分词器中
    def add_tokens_tokenizer(tokenizer, all_list):
        add = []
        for word in all_list:
            if word in tokenizer.vocab:
                pass
            else:
                print(f"'{word}' is not in the BERT vocabulary.")
                add.append(word)
        num_added_toks = tokenizer.add_tokens(add)
        print('Now we have added', num_added_toks, 'tokens')
        return tokenizer

    tokenizer = add_tokens_tokenizer(tokenizer, all_list)
    
    
    train_loader, val_loader, test_loader = give_dataset(config.dataset.T45, tokenizer)
    
    # dataset = CholecT50( 
    #         dataset_dir=data_dir, 
    #         dataset_variant=dataset_variant,
    #         test_fold=kfold,
    #         augmentation_list=data_augmentations,
    #         )
    # train_dataset, val_dataset, test_dataset = dataset.build()
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    
    for batch, (img, txt,(y1, y2, y3, y4)) in enumerate(train_loader):
            img, txt,y1, y2, y3, y4 = img.cuda(), txt.cuda(),y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            print(img.shape)
            print(txt.shape)
            print(y1.shape)
            print(y2.shape)
            print(y3.shape)
            print(y4.shape)