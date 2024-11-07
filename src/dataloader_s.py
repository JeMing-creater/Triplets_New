import os
import json
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class CholecT45():
    def __init__(self,
                 dataset_dir,
                 threshold,image_size=[224,224],
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                 augmentation_list_s=['original']):

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
        self.threshold = threshold
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
            'original': self.no_augumentation
        }
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])

        # 小类别的增强
        self.augmentations_s = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.augmentation_list_s = []
        for aug in augmentation_list_s:
            self.augmentation_list_s.append(self.augmentations_s[aug])

        trainform_s, trainform, testform = self.transform()

        self.build_train_dataset(trainform, trainform_s)
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
        op_train_s = [transforms.Resize((self.image_size[0], self.image_size[1]))] + self.augmentation_list_s + [transforms.Resize((self.image_size[0], self.image_size[1])),
                                                                                   transforms.ToTensor(), normalize, ]
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        trainform_s = transforms.Compose(op_train_s)
        return trainform_s, trainform, testform

    def build_train_dataset(self, transform, transform_s):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T45_s(img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          threshold=self.threshold,
                          transform=transform,
                          transform_s=transform_s
                          )
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)  # 从列表变成一个数据集对象

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T45(img_dir=os.path.join(self.dataset_dir, 'data', video),
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
            dataset = T45(img_dir=os.path.join(self.dataset_dir, 'data', video),
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
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))#从 triplet_labels 中获取当前样本的 ID（在第一列），将其转换为字符串，确保它是 6 位数字，并用零填充。
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

        return image, (tool_label, verb_label, target_label, triplet_label)

class T45_s(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, threshold, transform=None,
                 transform_s=None, target_transform=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',')
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',')
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.transform_s = transform_s
        self.target_transform = target_transform
        self.threshold = threshold
        self.binary_100_list = self.find_small_categories(self.threshold)




    def update_list(self,indices):
        # 创建一个长度为100的列表，初始值为0
        result_list = [0] * 100

        # 遍历输入的索引列表
        for index in indices:
            # 确保索引在有效范围内
            if 0 <= index < 100:
                result_list[index] = 1

        return result_list



    def find_small_categories(self,threshold):
        count_list = [8798, 4623, 457, 268, 38, 146, 278, 378, 2826, 1406, 90, 161, 53, 218, 4163, 376, 273, 2284,
                      25134, 14795, 1729, 14126, 614, 37, 479, 79, 16, 98, 113, 480, 360, 2228, 68, 60,
                      168, 149, 62, 239, 285, 0, 8, 95, 81, 8, 9, 58, 138, 91, 10, 16, 40, 27, 123, 182,
                      161, 0, 0, 21, 2042, 6176, 4054, 12764, 18466, 452, 206, 364, 161, 93, 21, 192, 691,
                      418, 38, 90, 56, 12, 100, 45, 37, 440, 1384, 912, 9, 709, 2288, 68, 31, 29, 100,
                      59, 361, 14, 150, 35, 731, 863, 3057, 2239, 1211, 283]

        small_categories = []  # 初始化空列表，用于存储小于阈值的索引
        for index, value in enumerate(count_list):
            if value < threshold:  # 检查值是否小于阈值
                small_categories.append(index)  # 将索引添加到列表中

        binary_100_list = self.update_list(small_categories)
        return binary_100_list




    def convert_to_list(self,triplet_label):
        # 直接将 numpy 数组转换为列表
        return triplet_label.tolist()



    def check_mark_in_categories(self,triplet_label, binary_100_list):
        # 检查两个列表的长度是否相等
        if len(triplet_label) != len(binary_100_list):
            raise ValueError("两者列表长度必须相同")
        # 判断是否有同一位置同时为1
        is_small = any(triplet_label[i] == 1 and binary_100_list[i] == 1 for i in range(len(triplet_label)))
        return int(is_small)  # 返回1或0

    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, index):



        triplet_label = self.triplet_labels[index, 1:]
        triplet_label=self.convert_to_list(triplet_label)


        # 调用 find_first_one_index 函数

        is_samll = self.check_mark_in_categories(triplet_label, self.binary_100_list)






        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(
            6))  # 从 triplet_labels 中获取当前样本的 ID（在第一列），将其转换为字符串，确保它是 6 位数字，并用零填充。
        img_path = os.path.join(self.img_dir, basename)
        image = Image.open(img_path)
        if  is_samll:
            image = self.transform_s(image)
        else:
            image = self.transform(image)
        '''
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
        '''
        triplet_label = self.triplet_labels[index, 1:]
        return image, (tool_label, verb_label, target_label, triplet_label)


class CholecT50():
    def __init__(self,
                 dataset_dir,image_size=[224,224],
                 dataset_variant="cholect50-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                 normalize=True,
                 m=3):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int,
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label, phase_label))
        """
        self.image_size = image_size
        self.normalize = normalize
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits (recommended)",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
            "cholect50-subset": "specially created for EDU4SDS summer school"
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
        op_test = [transforms.Resize((self.image_size[0], self.image_size[1])), transforms.ToTensor(), ]
        op_train = [transforms.Resize((self.image_size[0], self.image_size[1]))] + self.augmentation_list + [transforms.Resize((self.image_size[0], self.image_size[1])),
                                                                               transforms.ToTensor()]
        if self.normalize:
            op_test.append(normalize)
            op_train.append(normalize)
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
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'videos', video),
                          label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'videos', video),
                          label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)
        # self.val_dataset = iterable_dataset

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(img_dir=os.path.join(self.dataset_dir, 'videos', video),
                          label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform,
                          m=self.m)
            iterable_dataset.append(dataset)
        self.test_dataset = ConcatDataset(iterable_dataset)
        # self.test_dataset = iterable_dataset

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)


class T50(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_transform=None, m=3):
        label_data = json.load(open(label_file, "rb"))
        self.label_data = label_data["annotations"]
        self.frames = list(self.label_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.m = m

    def __len__(self):
        return len(self.frames)

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
        indices = list(
            map(lambda x: x + int(current_frame_id) if x + int(current_frame_id) > 0 else 0, np.arange(-self.m, 1)))
        # print("indices >> ", indices)

        labels = self.label_data[current_frame_id]
        images = [Image.open(os.path.join(self.img_dir, f"{int(i):06}.png")) for i in indices]
        labels = self.get_binary_labels(labels)

        if self.transform:
            images = list(map(lambda x: self.transform(x), images))
        if self.target_transform:
            labels = self.target_transform(labels)

        return np.stack(images, axis=0), (labels[0], labels[1], labels[2], labels[3])


def give_dataset(config):
    dataset_choose = config.trainer.dataset
    if dataset_choose == 'T45':
        config = config.dataset.T45
        dataset = CholecT45(
            image_size=config.image_size,
            dataset_dir=config.data_dir,
            threshold=config.threshold,
            dataset_variant=config.dataset_variant,
            test_fold=config.kfold,
            augmentation_list=config.data_augmentations,
            augmentation_list_s=config.data_augmentations_s
        )
        train_dataset, val_dataset, test_dataset = dataset.build()

        # 创建训练数据加载器
        train_dataloader = DataLoader(
            train_dataset,  # 要加载的数据集
            batch_size=config.batch_size,  # 每个批次的样本数量，由配置文件指定
            shuffle=True,  # 在每个 epoch 开始时随机打乱数据，防止模型过拟合
            prefetch_factor=3 * config.batch_size,  # 每个 worker 预先加载 3 个批次的数据，加快数据加载速度
            num_workers=config.num_workers,  # 使用的子进程数量，用于并行加载数据，减少数据加载的瓶颈
            pin_memory=config.pin_memory,  # 是否将数据加载到固定内存，以提高 GPU 训练时的数据传输速度
            persistent_workers=config.persistent_workers,  # 是否在每个 epoch 之间保持 worker 进程，避免频繁创建和销毁进程的开销
            drop_last=config.drop_last  # 如果数据集的样本数不能被 batch_size 整除，是否丢弃最后一个不完整的批次
        )

        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    prefetch_factor=3 * config.batch_size, num_workers=config.num_workers,
                                    pin_memory=config.pin_memory, persistent_workers=config.persistent_workers,
                                    drop_last=config.drop_last)

        # test data set is built per video, so load differently
        test_dataloaders = []
        for video_dataset in test_dataset:
            test_dataloader = DataLoader(video_dataset, batch_size=config.batch_size, shuffle=False,
                                         prefetch_factor=3 * config.batch_size, num_workers=config.num_workers,
                                         pin_memory=config.pin_memory, persistent_workers=config.persistent_workers,
                                         drop_last=config.drop_last)
            test_dataloaders.append(test_dataloader)

        return train_dataloader, val_dataloader, test_dataloader





    elif dataset_choose == 'T50':
        config = config.dataset.T50
        dataset = CholecT50(
            image_size=config.image_size,
            dataset_dir=config.data_dir,
            dataset_variant=config.dataset_variant,
            test_fold=config.kfold,
            augmentation_list=[config.data_augmentations.pop()],
            normalize=True,
            m=config.m
        )

        train_dataset, val_dataset, test_dataset = dataset.build()

        # create dataloader for train data
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4 * config.batch_size,
            persistent_workers=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=4 * config.batch_size,
            persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=4 * config.batch_size,
            persistent_workers=True
        )
        # val_dataloader = []
        # records = dataset.val_records
        # for i, video_ds in enumerate(val_dataset):
        #     loader = DataLoader(
        #                     video_ds,
        #                     batch_size=config.batch_size,
        #                     num_workers=config.num_workers,
        #                     shuffle=False,
        #                     pin_memory=True,
        #                     prefetch_factor=4*config.batch_size,
        #                     persistent_workers=True
        #                 )
        #     val_dataloader.append((records[i], loader))

        # test_dataloader = []
        # records = dataset.test_records
        # for i, video_ds in enumerate(test_dataset):
        #     loader = DataLoader(
        #                     video_ds,
        #                     batch_size=config.batch_size,
        #                     num_workers=config.num_workers,
        #                     shuffle=False,
        #                     pin_memory=True,
        #                     prefetch_factor=4*config.batch_size,
        #                     persistent_workers=True
        #                 )
        #     test_dataloader.append((records[i], loader))
        return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    # config setting
    config = EasyDict(
        yaml.load(open('D:\服务器代替\config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))


    train_loader, val_loader, test_loader = give_dataset(config)

    # dataset = CholecT45(
    #         dataset_dir=data_dir,
    #         dataset_variant=dataset_variant,
    #         test_fold=kfold,
    #         augmentation_list=data_augmentations,
    #         )
    # train_dataset, val_dataset, test_dataset = dataset.build()
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)

    for batch, (img, (y1, y2, y3, y4)) in enumerate(train_loader):
        img, y1, y2, y3, y4 = img, y1, y2, y3, y4
