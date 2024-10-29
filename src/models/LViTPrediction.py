'''
Description     : LViT Localized-Answering model
Paper           : Surgical-VQLA: Transformer with Gated Vision-Language Embedding for 
                  Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBERT ResMLP model from 
                  Surgical VQA (https://github.com/lalithjets/Surgical_VQA) and timm/models 
                  (https://github.com/rwightman/pytorch-image-models/tree/master/timm/models).
'''
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import clip
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer, CLIPModel, CLIPProcessor
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel, MedCLIPProcessor, PromptClassifier
from timm import create_model
# from models.GatedLanguageVisualEmbedding import VisualBertEmbeddings
# from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ImageProcess(nn.Module):
    def __init__(self, image_size, output_dim):
        super().__init__()
        self.flatten = nn.Flatten(2)  # 从第1维开始拉平，保留批次维度
        # self.linear1 = nn.Linear(image_size[0]*image_size[1], image_size[0]*image_size[1]//2)
        # self.linear2 = nn.Linear(image_size[0]*image_size[1]//2, output_dim)
        self.linear = nn.Linear(image_size[0]*image_size[1], output_dim)
        
        
    def forward(self, x):
        x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.linear(x)
        return x

class VisualBertEmbeddings(nn.Module):
    def __init__(self, config, max_size=40):
        super().__init__()
        self.config = config
        self.max_size = max_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        # self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        # self.visual_projection = nn.Linear(448, config.hidden_size)
        self.gated_linear = GatedMultimodalLayer(config.hidden_size*max_size, config.hidden_size*3, config.hidden_size*max_size)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        #  ================================================================
        # visual_embeds = self.visual_projection(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

        embeddings = torch.flatten(embeddings, start_dim=1, end_dim=-1)
        visual_embeddings = torch.flatten(visual_embeddings, start_dim=1, end_dim=-1)        
        embeddings = self.gated_linear(embeddings, visual_embeddings)  
        embeddings = torch.reshape(embeddings, (-1, self.max_size, self.config.hidden_size))
      
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2


'''
VisualBert Classification Model
'''
class LViTPrediction(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, max_size, image_size, num_tool=6, num_verb=10, num_target=15 ):
        super(LViTPrediction, self).__init__()

        self.config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        # self.config.visual_embedding_dim = 512
        self.config.vocab_size = vocab_size 
        self.config.num_hidden_layers = layers
        self.config.num_attention_heads = n_heads        

        self.process = ImageProcess(image_size, config.hidden_size)
        self.embeddings = VisualBertEmbeddings(config = self.config, max_size=max_size)
        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        self.tool = nn.Linear(self.config.hidden_size, num_tool)
        self.verb = nn.Linear(self.config.hidden_size, num_verb)
        self.target = nn.Linear(self.config.hidden_size, num_target)
        # self.bbox_embed = MLP(self.config.hidden_size, self.config.hidden_size, 4, 3)
    
    def get_classfiy(self, inputs, class_target = 'tool'):
        # Encoder output
        embedding_output = self.embeddings(
            input_ids = inputs['input_ids'].to(device),
            token_type_ids = inputs['token_type_ids'].to(device),
            position_ids = None,
            inputs_embeds = None,
            visual_embeds = inputs['visual_embeds'].to(device),
            visual_token_type_ids = inputs['visual_token_type_ids'].to(device),
            image_text_alignment = None,
        ) #[1, 56, 768]
        
        outputs = self.vit.blocks(embedding_output)
        outputs = self.vit.norm(outputs)
        outputs = outputs.mean(dim=1)             
        
        # classification layer 
        if class_target == 'tool':
            classification_outputs = self.tool(outputs)
        elif class_target == 'verb':
            classification_outputs = self.verb(outputs)
        else:
            classification_outputs = self.target(outputs)
        return classification_outputs
    
    def forward(self, inputs1, inputs2, inputs3, visual_embeds):
        # prepare visual embedding
        # append visual features to text
        visual_embeds = self.process(visual_embeds)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        
        inputs1.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        })
        inputs2.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        })
        inputs3.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        })
        tool = self.get_classfiy(inputs1, class_target = 'tool')
        verb = self.get_classfiy(inputs2, class_target = 'verb')
        target = self.get_classfiy(inputs3, class_target = 'target')
              
        return tool, verb, target


class VQA_Classifiy(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, tokenizer, layers=6, n_heads=8, image_size=(256,448), max_size=40, num_tool=6, num_verb=10, num_target=15, num_triplet=100,):
        super(VQA_Classifiy, self).__init__()
        self.tokenizer = tokenizer
        self.VQA_model = LViTPrediction(vocab_size=len(tokenizer), layers=layers, n_heads=n_heads, max_size=max_size, image_size=image_size, num_tool=num_tool, num_verb=num_verb, num_target=num_target)
    
    def forward(self, inputs):
        questions1 = []
        questions2 = []
        questions3 = []
        for i in range(0, inputs.size()[0]):
            questions1.append('What surgical instrument is the doctor currently using?')
            questions2.append('What surgical action is the doctor currently performing?')
            questions3.append('What target is the doctor currently handling?')
            
        q1 = self.tokenizer(questions1, return_tensors="pt", padding="max_length", max_length=40)
        q2 = self.tokenizer(questions2, return_tensors="pt", padding="max_length", max_length=40)
        q3 = self.tokenizer(questions3, return_tensors="pt", padding="max_length", max_length=40)
        
        tool, verb, target = self.VQA_model(q1, q2, q3, inputs)
        
        return tool, verb, target


if __name__ == '__main__':
    # instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
    # target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'othertarget']       
    # verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'otherverb']      
    # all_list = instrument_list + target_list + verb_list
    
    # tokenizer = BertTokenizer.from_pretrained('C:\\Users\\90512\\Desktop\\Surgical-VQLA-main\\dataset\\bertvocab\\v2\\bert-EndoVis-18-VQA\\')
    
    # model = VQA_Classifiy(tokenizer = tokenizer).to(device)
    
    visual_features = torch.rand(2, 3, 256, 448).to(device)
    
    # tool, verb, target = model(visual_features)
    
    text = 'In this picture, the doctor uses grassper to perform grassp actions and deal with galbladder.'  
    image = Image.open('C:\\Users\\90512\\Desktop\\test_root\\1.png').convert("RGB")
    
    # encoding = processor(text, return_tensors=None, **{})
    
    
    # # tokenizer = add_tokens_tokenizer(tokenizer, all_list)
    # model = LViTPrediction(vocab_size=len(tokenizer), layers=6, n_heads=8, image_size=(256,448), max_size=40, num_class = 100).to(device)
    
    # questions = ['What operation is the doctor performing?', ]
    # inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=40)
    
    # # image = Image.open('C:\\Users\\90512\\Desktop\\test_root\\1.png').convert('RGB')

    # # processor = AutoProcessor.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    # visual_features = torch.rand(1, 3, 256, 448).to(device)
    
    # classification_outputs = model(inputs, visual_features)
    # print(classification_outputs)
    
