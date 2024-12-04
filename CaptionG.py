
import os
import re
import json
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
import yaml
import math
import numbers
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from flask import Flask, jsonify
# from net import TransformerBlock as Restormer
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model
from easydict import EasyDict


# 初始化模型和分词器
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

class T45GenerateWord():
    def __init__(self,dataset_dir, save_path = 'words'):
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
            'grasper.',
            'bipolar.',
            'hook.',
            'scissors.',
            'clipper.',
            'irrigator.'
        ]
        self.dataset_dir = dataset_dir
        self.model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        self.records = self.get_max_num()
        self.save_path = dataset_dir + '/'+ save_path
        self.create_path()
    
    def create_path(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"'{self.save_path}' have been created. ")
        else:
            print(f"'{self.save_path}' is exist.")
    
    def get_max_index(self, file_path):
        max_index = -1  # 初始化最大索引为-1，假设文件中的索引从0开始
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 假设每行的格式是 "index text"
                parts = line.strip().split(',', 1)  # 分割索引和文本
                if len(parts) == 2 and parts[0].isdigit():  # 确保第一部分是数字
                    index = int(parts[0])  # 将索引转换为整数
                    if index > max_index:
                        max_index = index  # 更新最大索引

        return max_index
    
    def test_word_index(self, path, video):
        triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video))
        triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        exist_data_max_index = self.get_max_index(path)
        if exist_data_max_index == (len(triplet_labels)-1):
            return True, exist_data_max_index
        else:
            return False, exist_data_max_index
    
    def crate_txt(self, path, video):
        if os.path.exists(path):
            flig, exist_data_max_index = self.test_word_index(path, video)
            return flig, exist_data_max_index
            
        
        # 创建一个新的TXT文件
        with open(path, 'w') as file:
            print(f"'{path}' have been created.")
        return False, -1
    
    def get_max_num(self):
        # 存储找到的文件夹编号
        folder_all = []
        
        # 列出路径下的所有文件夹
        for folder_name in os.listdir(self.dataset_dir+'/data/'):
            folder_all.append(folder_name)
            
        return folder_all

    def traversal_triplet(self, triplet_labels, save_txt_path, exist_data_max_index):
        # labels = []
        print(f'Start from {exist_data_max_index}')
        for index in range(0, len(triplet_labels)):
            if index <= exist_data_max_index:
                continue
            label = triplet_labels[index, 1:]
            text_labels = [item for label, item in zip(label, self.labels) if label == 1]
            
            if text_labels == []:
                text = 'The doctor has not taken any action at the moment.'
            else:
                add_text = ''
                for l in text_labels:
                    add_text = add_text + ' ' + l
                add_text = add_text 
                # prompt = f"I am describing a surgical picture of a gallbladder removal operation. Here are some specific actions in the picture: [{add_text}] Please help me understand and describe the entire content of the picture in one sentence."
                # prompt = f"During the cholecystectomy, the doctor performed the following actions or simply held tools:\n{add_text}Please summarize the action text I provided in English, or just tell me what tools the doctor was using, in no more than 200 words, and without Chinese characters. "
                prompt = f"During the cholecystectomy, the doctor is performing the following actions or juse holding up a tool: [{add_text}] Summarize the doctor's actions or state which tool the doctor is using in English. It is worth noting that if you are informed of actions and tool descriptions in more than one sentence, please help me summarize it into one sentence (no more than 200 words). If you are unable to understand the information I am sending, only reply to the text content within []"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                in_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([in_text], return_tensors="pt").to(device)

                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            with open(save_txt_path, 'a', encoding='utf-8') as output_file:
                # 逐行读取原始文件
                output_file.write(f"{index},{text}\n")
                
            print(text)
            # labels.append(text)
        # return labels
    
    
    def generate_word_save(self):
         for video in self.records:
             print(f'Generate {video} data!')
             save_txt_path = self.save_path + f'/{video}.txt'
             flig, exist_data_max_index = self.crate_txt(save_txt_path, video)
             triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video))
             triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
            #  self.traversal_triplet(triplet_labels, save_txt_path, exist_data_max_index)
             if flig != True:
                self.traversal_triplet(triplet_labels, save_txt_path, exist_data_max_index)
             
class T50GenerateWord():
    def __init__(self,dataset_dir, save_path = 'words'):
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
            'grasper.',
            'bipolar.',
            'hook.',
            'scissors.',
            'clipper.',
            'irrigator.'
        ]
        self.dataset_dir = dataset_dir
        self.model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        self.save_path = dataset_dir + '/'+ save_path
        self.records = self.get_max_num()
        self.create_path()
        
    def create_path(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"'{self.save_path}' have been created. ")
        else:
            print(f"'{self.save_path}' is exist.")
    
    def get_max_num(self):
        # 存储找到的文件夹编号
        folder_all = []
        
        # 列出路径下的所有文件夹
        for folder_name in os.listdir(self.dataset_dir+'/videos/'):
            folder_all.append(folder_name)
            
        return folder_all
    
    def get_max_index(self, file_path):
        max_index = -1  # 初始化最大索引为-1，假设文件中的索引从0开始
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 假设每行的格式是 "index text"
                parts = line.strip().split(',', 1)  # 分割索引和文本
                if len(parts) == 2 and parts[0].isdigit():  # 确保第一部分是数字
                    index = int(parts[0])  # 将索引转换为整数
                    if index > max_index:
                        max_index = index  # 更新最大索引

        return max_index
    
    def test_word_index(self, path, video, label_data):
        # triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video))
        # triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',')
        # exist_data_max_index = self.get_max_index(path)
        exist_index = []
        # label_data = list(label_data.keys())
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',', 1)  
                exist_index.append(parts[0])
        
        if len(exist_index) == len(label_data):
            return True, label_data
        else:
            add_list = []     
            for key in label_data:
                if key not in exist_index:
                    add_list.append(key)
            return False, add_list
        # if exist_data_max_index == (len(triplet_labels)-1):
        #     return True, exist_data_max_index
        # else:
        #     return False, exist_data_max_index
    
    def crate_txt(self, path, video, label_data):
        label_data = list(label_data.keys())
        if os.path.exists(path):
            flig, exist_data_max_index = self.test_word_index(path, video, label_data)
            return flig, exist_data_max_index
            
        # 创建一个新的TXT文件
        with open(path, 'w') as file:
            print(f"'{path}' have been created.")
        return False, label_data
    
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
    
    def traversal_triplet(self, triplet_labels, save_txt_path, exist_data_max_index):
        # labels = []
        print(f'Need to add {exist_data_max_index}')
        frames = list(triplet_labels.keys())
        for index in range(0, len(frames)):
            current_frame_id = frames[index]
            if current_frame_id not in exist_data_max_index:
                continue
            
            # indices = list(map(lambda x: x+int(current_frame_id) if x+int(current_frame_id) > 0 else 0, np.arange(-self.m, 1)))
            
            labels = triplet_labels[current_frame_id]
            
            label = self.get_binary_labels(labels)[3]
            
            # label = labels[index, 1:]
            text_labels = [item for label, item in zip(label, self.labels) if label == 1]
            
            if text_labels == []:
                text = 'The doctor has not taken any action at the moment.'
            else:
                add_text = ''
                for l in text_labels:
                    add_text = add_text + ' ' + l
                add_text = add_text 
                # prompt = f"I am describing a surgical picture of a gallbladder removal operation. Here are some specific actions in the picture: [{add_text}] Please help me understand and describe the entire content of the picture in one sentence."
                # prompt = f"During the cholecystectomy, the doctor performed the following actions or simply held tools:\n{add_text}Please summarize the action text I provided in English, or just tell me what tools the doctor was using, in no more than 200 words, and without Chinese characters. "
                prompt = f"During the cholecystectomy, the doctor is performing the following actions or juse holding up a tool: [{add_text}] Summarize the doctor's actions or state which tool the doctor is using in English. It is worth noting that if you are informed of actions and tool descriptions in more than one sentence, please help me summarize it into one sentence (no more than 200 words). If you are unable to understand the information I am sending, only reply to the text content within []"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                in_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([in_text], return_tensors="pt").to(device)

                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            with open(save_txt_path, 'a', encoding='utf-8') as output_file:
                # 逐行读取原始文件
                output_file.write(f"{current_frame_id},{text}\n")
                
            print(text)
            # labels.append(text)
        # return labels
    
    def generate_word_save(self):
        for video in self.records:
            print(f'Generate {video} data!')
            save_txt_path = self.save_path + f'/{video}.txt'
            label_data = json.load(open(os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)), "rb"))["annotations"]
            flig, exist_data_max_index = self.crate_txt(save_txt_path, video, label_data)
            # exist_data_max_index = -1
            if flig != True:
                self.traversal_triplet(label_data, save_txt_path, exist_data_max_index)

if __name__ == '__main__':
    device = 'cuda:3'
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

    if config.trainer.dataset == 'T45':
        # TODO: caption generation
        g = T45GenerateWord(dataset_dir=config.dataset.T45.data_dir)
        folder_all = g.generate_word_save()
    else:
        g = T50GenerateWord(dataset_dir=config.dataset.T50.data_dir)
        folder_all = g.generate_word_save()
    