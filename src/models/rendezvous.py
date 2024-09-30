#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Nwoye Chinedu Innocent
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as basemodels
import torchvision.transforms as transforms


OUT_HEIGHT = 8
OUT_WIDTH  = 14


#%% Model Rendezvous
class Rendezvous(nn.Module):
    def __init__(self, basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, hr_output=False, use_ln=False):
       #hr_output 参数用于控制基础模型（BaseModel）是否使用高分辨率输出。
       #参数控制是否在解码器（Decoder）中使用层归一化（Layer Normalization）。
        super(Rendezvous, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output)
        self.decoder = Decoder(layer_size, d_model, num_heads, num_triplet, use_ln=use_ln)    
     
    def forward(self, inputs):
        #先将input输入进去,得到编码器输出的enc_i, enc_v, enc_t, enc_ivt,然后将这四个丢进decode
        enc_i, enc_v, enc_t, enc_ivt = self.encoder(inputs)
        dec_ivt = self.decoder(enc_i, enc_v, enc_t, enc_ivt)
        return enc_i, enc_v, enc_t, dec_ivt


#%% Triplet Components Feature Encoder
class Encoder(nn.Module):
    def __init__(self, basename='resnet18', num_tool=6,  num_verb=10, num_target=15, num_triplet=100, hr_output=False):
        super(Encoder, self).__init__()
        depth = 64 if basename == 'resnet18' else 128
        self.basemodel  = BaseModel(basename, hr_output)
        self.wsl        = WSL(num_tool, depth) #弱监督定位
        self.cagam      = CAGAM(num_tool, num_verb, num_target)
        self.bottleneck = Bottleneck(num_triplet)
        
    def forward(self, x):
        high_x, low_x = self.basemodel(x)#接收返回来的高分辨率特征和低分辨率特征
        # print("经过resnet特征提取器后:")
        # print("high_x.shape",high_x.shape), "       ","low_x.shape",print(low_x.shape)
        enc_i         = self.wsl(high_x)
        # print("经过这个wsl之后:")

        # print("enc_i[0].shape",enc_i[0].shape)
        # print("enc_i[1].shape",enc_i[1].shape)
        enc_v, enc_t  = self.cagam(high_x, enc_i[0])
        # print("经过这个CAGAM之后:")

        # print("enc_v[0].shape", enc_v[0].shape)
        # print("enc_v[1].shape", enc_v[1].shape)

        # print("enc_t[0].shape", enc_t[0].shape)
        # print("enc_t[1].shape", enc_t[1].shape)
        enc_ivt       = self.bottleneck(low_x)
        # print("bottleneck的输入low_x形状是",low_x.shape, "输出的enc_ivt的形状是",enc_ivt.shape)
        return enc_i, enc_v, enc_t, enc_ivt


#%% MultiHead Attention Decoder
class Decoder(nn.Module):
    def __init__(self, layer_size, d_model, num_heads, num_class=100, use_ln=False):
        super(Decoder, self).__init__()        
        self.projection = nn.ModuleList([Projection(num_triplet=num_class, out_depth=d_model) for i in range(layer_size)])
        self.mhma       = nn.ModuleList([MHMA(num_class=num_class, depth=d_model, num_heads=num_heads, use_ln=use_ln) for i in range(layer_size)])
        self.ffnet      = nn.ModuleList([FFN(k=layer_size-i-1, num_class=num_class, use_ln=use_ln) for i in range(layer_size)])
        self.classifier = Classifier(num_class)
        
    def forward(self, enc_i, enc_v, enc_t, enc_ivt):
        i=1
        X = enc_ivt.clone()
        for P, M, F in zip(self.projection, self.mhma, self.ffnet):
            X = P(enc_i[0], enc_v[0], enc_t[0], X)
            # print("第",i,"次这个projection操作后x元组的形状是",len(X))
            X = M(X)
            # print("第", i, "次这个MHMA操作后x的形状是", X.shape)
            X = F(X)
            # print("第", i, "次这个FFN操作后x的形状是", X.shape)
            i+=1
        logits = self.classifier(X)
        # print("经过分类器后这个logits的形状是:",logits.shape)
        return logits


#%% Feature extraction backbone
class BaseModel(nn.Module):   
    def __init__(self, basename='resnet18', hr_output=False, *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {} 
        if basename == 'resnet18':
            self.basemodel      = basemodels.resnet18(pretrained=True)     
            if hr_output: self.increase_resolution()#调整增加模型的分辨率。减小conv1 层的步幅和downsample
            self.basemodel.layer1[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            #注册前向钩子 get_activation，以便在 ResNet50 的特定层（layer1[2].bn2 和 layer4[2].bn2）前向传播时，捕获并保存这些层的输出特征。
        if basename == 'resnet50':
            self.basemodel      = basemodels.resnet50(pretrained=True)
            self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
        # # 由于使用hook，损失回传时将不更新fc，在多卡训练环境下，将导致错误，因此冻结该部分参数。
        for param in self.basemodel.fc.parameters():
                param.requires_grad = False
                
    def increase_resolution(self):  
        global OUT_HEIGHT, OUT_WIDTH  
        self.basemodel.layer3[0].conv1.stride = (1,1)
        self.basemodel.layer3[0].downsample[0].stride=(1,1)  
        self.basemodel.layer4[0].conv1.stride = (1,1)
        self.basemodel.layer4[0].downsample[0].stride=(1,1)
        OUT_HEIGHT *= 4
        OUT_WIDTH  *= 4#改变输出的高和宽
        print("using high resolution output ({}x{})".format(OUT_HEIGHT,OUT_WIDTH))  #打印新的分辨率

    def get_activation(self, layer_name):
        def hook(module, input, output):
            self.output_feature[layer_name] = output
        return hook#这里返回的是这个hook函数给调用这个get_activation的时候调用
    
    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']
    #返回低级特征和高级特征

     
#%% Weakly-Supervised localization
class WSL(nn.Module):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=depth, kernel_size=3, padding=1)
        self.cam   = nn.Conv2d(in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn    = nn.BatchNorm2d(depth)
        self.gmp   = nn.AdaptiveMaxPool2d((1,1))# 自适应最大池化层
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam     = self.cam(feature)
        logits  = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits
 

#%% Unfiltered Bottleneck layer
class Bottleneck(nn.Module):#输入low_x的64个通道,输出三元组100个类别
    def __init__(self, num_class):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, stride=(2,2), kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn1   = nn.BatchNorm2d(256)
        self.bn2   = nn.BatchNorm2d(num_class)
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.elu(feature)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.elu(feature)
        return feature


#%% Class Activation Guided Attention Mechanism
class CAGAM(nn.Module):    
    def __init__(self, num_tool, num_verb, num_target, in_depth=512):
        super(CAGAM, self).__init__()        
        out_depth               = num_tool
        
        #分别生成动作和动作-工具的键值对的对应卷积操作
        self.verb_context       = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)        
        self.verb_query         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_key           = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key      = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_cmap          = nn.Conv2d(in_channels=out_depth, out_channels=num_verb, kernel_size=1)

        # 分别生成目标和目标-工具的键值对的对应卷积操作
        self.target_context     = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)     
        self.target_query       = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_query  = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_key         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_key    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_cmap        = nn.Conv2d(in_channels=out_depth, out_channels=num_target, kernel_size=1)


        self.gmp       = nn.AdaptiveMaxPool2d((1,1))
        self.elu       = nn.ELU()    
        self.soft      = nn.Softmax(dim=1)    
        self.flat      = nn.Flatten(2,3)  
        self.bn1       = nn.BatchNorm2d(out_depth)
        self.bn2       = nn.BatchNorm2d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm2d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm2d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm2d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth)
        self.bn10      = nn.BatchNorm2d(out_depth) 
        self.bn11      = nn.BatchNorm2d(out_depth) 
        self.bn12      = nn.BatchNorm2d(out_depth)        
        self.encoder_cagam_verb_beta   = torch.nn.Parameter(torch.randn(1))
        self.encoder_cagam_target_beta = torch.nn.Parameter(torch.randn(1))          
        #是两个可学习的参数，它们在模型训练过程中会被优化。它们的主要作用是调整和控制模型中注意力机制的输出
    def get_verb(self, raw, cam):
        x  = self.elu(self.bn1(self.verb_context(raw)))#通道变成工具的分类数
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        #后续自动推断batch的大小

        #注意力机制的一般流程
        q1 = self.elu(self.bn2(self.verb_query(x)))
        k1 = self.elu(self.bn3(self.verb_key(x)))
        w1 = self.flat(k1).matmul(self.flat(q1).transpose(-1,-2))        
        q2 = self.elu(self.bn4(self.verb_tool_query(cam)))
        k2 = self.elu(self.bn5(self.verb_tool_key(cam)))
        w2 = self.flat(k2).matmul(self.flat(q2).transpose(-1,-2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))#除根号Z
        attention = self.soft(attention)    #softmax归一化

        v = self.flat(z)#展开方便注意力机制
        e = (attention.matmul(v) * self.encoder_cagam_verb_beta).reshape(sh)#用学习参数调整
        e = self.bn6(e + z)
        #使用 BatchNorm2d 进行归一化，e + z 表示残差连接（即加入原始特征图 z）。
        cmap = self.verb_cmap(e)#一个卷积操作,也是将这个是输出的通道数设置为对应的类别数
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y  
    
    def get_target(self, raw, cam):
        x  = self.elu(self.bn7(self.target_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn8(self.target_query(x)))
        k1 = self.elu(self.bn9(self.target_key(x)))
        w1 = self.flat(k1).transpose(-1,-2).matmul(self.flat(q1))        
        q2 = self.elu(self.bn10(self.target_tool_query(cam)))
        k2 = self.elu(self.bn11(self.target_tool_key(cam)))
        w2 = self.flat(k2).transpose(-1,-2).matmul(self.flat(q2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         
        v = self.flat(z)
        e = (v.matmul(attention) * self.encoder_cagam_target_beta).reshape(sh)
        e = self.bn12(e + z)
        cmap = self.target_cmap(e)
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y
            
    def forward(self, x, cam):#调用这个类的时候传进来的东西就是左边这两个
        cam_v, logit_v = self.get_verb(x, cam)
        cam_t, logit_t = self.get_target(x, cam)
        return (cam_v, logit_v), (cam_t, logit_t)

 
#%% Projection function
class Projection(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128):
        super(Projection, self).__init__()

        self.ivt_value = nn.Conv2d(in_channels=num_triplet, out_channels=out_depth, kernel_size=1)
        self.i_value   = nn.Conv2d(in_channels=num_tool, out_channels=out_depth, kernel_size=1)        
        self.v_value   = nn.Conv2d(in_channels=num_verb, out_channels=out_depth, kernel_size=1)
        self.t_value   = nn.Conv2d(in_channels=num_target, out_channels=out_depth, kernel_size=1)

        self.ivt_query = nn.Linear(in_features=num_triplet, out_features=out_depth)

        self.dropout   = nn.Dropout(p=0.3)

        self.ivt_key   = nn.Linear(in_features=num_triplet, out_features=out_depth)
        self.i_key     = nn.Linear(in_features=num_tool, out_features=out_depth)
        self.v_key     = nn.Linear(in_features=num_verb, out_features=out_depth)
        self.t_key     = nn.Linear(in_features=num_target, out_features=out_depth)

        self.gap       = nn.AdaptiveAvgPool2d((1,1))
        self.elu       = nn.ELU()          
        self.bn1       = nn.BatchNorm1d(out_depth)
        self.bn2       = nn.BatchNorm1d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm1d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm1d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm1d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth) 
        
    def forward(self, cam_i, cam_v, cam_t, X):  
        q = self.elu(self.bn1(self.ivt_query(self.dropout(self.gap(X).squeeze(-1).squeeze(-1)))))  
        k = self.elu(self.bn2(self.ivt_key(self.gap(X).squeeze(-1).squeeze(-1))) )
        v = self.bn3(self.ivt_value(X)) 
        k1 = self.elu(self.bn4(self.i_key(self.gap(cam_i).squeeze(-1).squeeze(-1))) )
        v1 = self.elu(self.bn5(self.i_value(cam_i)) )
        k2 = self.elu(self.bn6(self.v_key(self.gap(cam_v).squeeze(-1).squeeze(-1))))
        v2 = self.elu(self.bn7(self.v_value(cam_v)) )
        k3 = self.elu(self.bn8(self.t_key(self.gap(cam_t).squeeze(-1).squeeze(-1))))
        v3 = self.elu(self.bn9(self.t_value(cam_t)))
        sh = list(v1.shape)
        v  = self.elu(F.interpolate(v, (sh[2],sh[3])))
        X  = self.elu(F.interpolate(X, (sh[2],sh[3])))
        return (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v))


#%% Multi-head of self and cross attention
class MHMA(nn.Module):
    def __init__(self, depth, num_class=100, num_heads=4, use_ln=False):
        super(MHMA, self).__init__()        
        self.concat = nn.Conv2d(in_channels=depth*num_heads, out_channels=num_class, kernel_size=3, padding=1)
        self.bn     = nn.BatchNorm2d(num_class)
        self.ln     = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        self.elu    = nn.ELU()    
        self.soft   = nn.Softmax(dim=1) 
        self.heads  = num_heads
        
    def scale_dot_product(self, key, value, query):
        dk        = torch.sqrt(torch.tensor(list(key.shape)[-2], dtype=torch.float32))
        affinity  = key.matmul(query.transpose(-1,-2))                        
        attn_w    = affinity / dk              
        attn_w    = self.soft(attn_w)
        attention = attn_w.matmul(value) 
        return attention
    
    def forward(self, inputs):
        (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v)) = inputs     
        query = torch.stack([q]*self.heads, dim=1) # [B,Head,D]
        query = query.unsqueeze(dim=-1) # [B,Head,D,1]        
        key   = torch.stack([k,k1,k2,k3], dim=1) # [B,Head,D]
        key   = key.unsqueeze(dim=-1) # [B,Head,D,1]        
        value = torch.stack([v,v1,v2,v3], dim=1) # [B,Head,D,H,W]
        dims  = list(value.shape) # [B,Head,D,H,W]
        value = value.reshape([-1,dims[1],dims[2],dims[3]*dims[4]])# [B,Head,D,HW]          
        attn  = self.scale_dot_product(key, value, query)  # [B,Head,D,HW]
        attn  = attn.reshape([-1,dims[1]*dims[2],dims[3],dims[4]]) # [B,DHead,H,W]
        mha   = self.elu(self.bn(self.concat(attn)))
        mha   = self.ln(mha + X.clone())  
        return mha

 
#%% Feed-forward layer
class FFN(nn.Module):#特征提取和增强
    def __init__(self, k, num_class=100, use_ln=False):
        super(FFN, self).__init__()
        def Ignore(x): return x
        self.conv1 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=1) 
        self.elu1  = nn.ELU() 
        self.elu2  = nn.ELU() if k>0 else Ignore     
        self.bn1   = nn.BatchNorm2d(num_class)    
        self.bn2   = nn.BatchNorm2d(num_class)
        self.ln    = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        
    def forward(self, inputs,):
        x  = self.elu1(self.bn1(self.conv1(inputs)))
        x  = self.elu2(self.bn2(self.conv2(x)))
        x  = self.ln(x + inputs.clone())
        return x

 
#%% Classification layer
class Classifier(nn.Module):
    def __init__(self, layer_size, num_class=100):
        super(Classifier, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1,1)) 
        self.mlp = nn.Linear(in_features=num_class, out_features=num_class)     
        
    def forward(self, inputs):
        x = self.gmp(inputs).squeeze(-1).squeeze(-1)
        y = self.mlp(x)
        return y

