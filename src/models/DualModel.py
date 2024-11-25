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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up5 = nn.ConvTranspose2d(in_channels // 16, in_channels // 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.head = nn.Conv2d(in_channels // 32, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.transpose(-1, -2)
        B, C, N = x.shape
        result = int(math.sqrt(N))
        x = x.reshape(B, C, result, result)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.head(x)
        return x
        

class PublicClassify(nn.Module):
    def __init__(self, n_features=1024, class_num=131, pretrained=True):
        super().__init__()
        self.head = nn.Linear(n_features, class_num)
        
    def forward(self, x):
        return self.head(x)

class TripletModel(nn.Module):
    def __init__(self, model_name, class_num=131, pretrained=True, mask_prob = 0.3):
        super().__init__()

        """
        Models class to return swin transformer models
        """
        self.mask_prob = mask_prob
        self.output_feature = {} 

        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # print(self.model)
        self.model.layers.register_forward_hook(self.get_activation('encoder feature'))
        # Get the number features in final embedding
        n_features = self.model.head.in_features

        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, class_num)
        
        self.decoder = Decoder(n_features, 3)
        
    def get_activation(self, layer_name):
        def hook(module, input: Tuple[torch.Tensor], output:torch.Tensor):
            self.output_feature[layer_name] = output
        return hook
    
    def get_mask_feature(self, x):
        batch_size, _, _, _ = x.shape
        mask = torch.rand(batch_size, 1, 1, 1) < self.mask_prob
        mask = mask.to(x.device)
        mask = mask.expand_as(x)
        # 应用掩码，将掩码位置的像素设置为 1 (0-1归一化后，最大值为1)
        masked_features = torch.where(mask, torch.full_like(x, 1), x)
        
        self.output_feature = {}
        _ = self.model(masked_features)
        encoder_feature = self.output_feature['encoder feature']
        return encoder_feature
    
    def forward(self, x):
        # first
        _ = self.model(x)
        feature_map = self.output_feature['encoder feature']
        
        x = self.model.head(self.model.norm(feature_map).mean(dim=1))
        
        out_image = self.decoder(feature_map)
        
        mask_feature = self.get_mask_feature(out_image)
        
        return x, feature_map, out_image, mask_feature



# class TripletModel(nn.Module):
#     def __init__(self, model_name, class_num=131, mask_prob=0.5,pretrained=True):
#         super().__init__()

#         """
#         Models class to return swin transformer models
#         """
#         self.output_feature = {} 

#         self.mask_prob = mask_prob
#         # Load the backbone
#         self.model1 = timm.create_model(model_name, pretrained=pretrained)
#         self.model1.layers.register_forward_hook(self.get_activation('encoder feature'))
#         self.model1.head.requires_grad = False
#         self.model1.norm.requires_grad = False
        
#         self.model2 = timm.create_model(model_name, pretrained=pretrained)
#         self.model2.layers.register_forward_hook(self.get_activation('momentum feature'))
#         self.model2.requires_grad = False
        
#         # Load the class_head
#         self.class_head = nn.Linear(in_features=1024, out_features=131, bias=True)
        
#         # Load decoder
#         self.decoder = Decoder(1024, 3)
        
#     def get_activation(self, layer_name):
#         def hook(module, input: Tuple[torch.Tensor], output:torch.Tensor):
#             self.output_feature[layer_name] = output
#         return hook
    
#     def random_crop_from_upsampled(self, x, scale_factor=2):
#         _, _, W, H = x.shape
#         tensor = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
#         _, _, new_W, new_H = tensor.shape

#         assert new_W >= W and new_H >= H

#         start_W = random.randint(0, new_W - W )
#         start_H = random.randint(0, new_H - H )

#         cropped_tensor = tensor[:, :, start_W:start_W + W, start_H:start_H + H]

#         return cropped_tensor
    
#     def get_encoder_feature(self, x):
#         _ = self.model1(x)
#         encoder_feature = self.output_feature['encoder feature']
#         return encoder_feature
    
#     def get_momentum_feature(self, x):
#         input_x = self.random_crop_from_upsampled(x, scale_factor=1.5)
#         _ = self.model2(input_x)
#         momentum_feature = self.output_feature['momentum feature']
#         return momentum_feature
    
#     def get_mask_feature(self, x):
#         mask = torch.rand_like(x) > self.mask_prob
#         masked_features = x * mask
#         self.output_feature = {}
#         _ = self.model1(masked_features)
#         encoder_feature = self.output_feature['encoder feature']
#         return encoder_feature
    
#     def forward(self, x):
#         # first
#         encoder_feature = self.get_encoder_feature(x)
#         momentum_feature = self.get_momentum_feature(x)
#         # second
#         encoder_class = self.class_head(torch.mean(encoder_feature, dim=1))
#         momentum_class = self.class_head(torch.mean(momentum_feature, dim=1))
#         # thrid
#         out_image = self.decoder(encoder_feature)
#         # fourth
#         out_image_feature = self.get_mask_feature(out_image)
        
#         return (encoder_feature, momentum_feature), (encoder_class, momentum_class), out_image, out_image_feature

if __name__ == '__main__':
    device = 'cuda:7'
    
    image = torch.randn(2, 3, 224, 224).to(device)
    
    model = TripletModel(model_name='swin_base_patch4_window7_224').to(device)
    
    output = model(image)[0]
    
    triple = output[:, :100]
    tool = output[:, 100:106]
    verb = output[:, 106:116]
    target = output[:, 116:]
    # print(output.shape)