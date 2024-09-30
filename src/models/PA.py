import os
import time
import math
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.nn import Linear
from functools import partial
import torch.nn.functional as F
import torchvision.models as basemodels
import torchvision.transforms as transforms
from timm.models.registry import register_model
from transformers import BertTokenizer, BertModel, MobileBertTokenizer, MobileBertModel,BertForMaskedLM
from timm.models.layers import DropPath, to_2tuple, make_divisible, trunc_normal_

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self,in_chans=3, embed_dims= [64, 128, 320, 512], **kwargs):
        super(pvt_v2_b2, self).__init__(
            in_chans=in_chans, patch_size=4, embed_dims=embed_dims, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class AttentionClassifier(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=4, num_class=100, i_num=6, t_num=15, v_num=10):
        super(AttentionClassifier, self).__init__() 
        
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(value_dim, key_dim)
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads

        assert self.head_dim * num_heads == key_dim, "key_dim must be divisible by num_heads"
        
        self.instrument_shape_layer = Linear(i_num * v_num, key_dim)
        self.target_shape_layer = Linear(t_num * v_num, key_dim)
        self.verb_shape_layer = Linear(v_num * i_num, key_dim)
        
        # self.gmp = nn.AdaptiveMaxPool2d((1,1)) 
        self.mlp = nn.Linear(in_features=key_dim, out_features=num_class)    
        
    def forward(self, inputs):
        instrument, target, verb = inputs
        i_num = instrument.size()[-1] # 6
        t_num = target.size()[-1] # 15
        v_num = verb.size()[-1] # 10
        
        query = instrument.repeat(1, 1, v_num).view(-1, i_num * v_num)
        # query = self.instrument_shape_layer(query)
        key = target.repeat(1, 1, v_num).view(-1, t_num * v_num)
        # key = self.target_shape_layer(key)
        value = verb.repeat(1, i_num, 1).view(-1, v_num * i_num)
        # value = self.verb_shape_layer(value)
        
        batch_size = query.size(0)

        # Project the query, key, and value matrices
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attention = F.softmax(scores, dim=-1)

        # Aggregate the values
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        out = self.mlp(out.squeeze(1))
        return out

class PA(nn.Module):
    def __init__(self, tokenizer, in_channel = 3, out_channel = 100, i_num = 6, t_num=15, v_num=10, num_heads=10, dims = [64, 128, 320, 512], model_dir='/root/.cache/huggingface/forget/pvt_v2_b3.pth'):
        super(PA, self).__init__()
        self.tokenizer = tokenizer
        
        # =========================== Text ===============================
        # self.model = MobileBertModel.from_pretrained("mobilebert-uncased")
        # self.testmodel = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.text_embeddings = self.model.embeddings
        self.text_encoder = self.model.encoder
        # self.text_pooler = self.model.pooler
        
        # self.instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'null_instrument']  
        self.instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
        self.target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'null_target']       
        self.verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'null_verb']      
        
        self.instrument_layer = Linear(self.model.config.hidden_size, len(self.instrument_list))
        self.target_layer = Linear(self.model.config.hidden_size, len(self.target_list))
        self.verb_layer = Linear(self.model.config.hidden_size, len(self.verb_list))
        
        # ========================== Image ===============================
        self.image_encoder = pvt_v2_b2(in_chans=in_channel, embed_dims=dims)
        if os.path.isfile(model_dir):
            save_model = torch.load(model_dir)
            model_dict = self.image_encoder.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.image_encoder.load_state_dict(model_dict)
        # self.image_layer = Linear((img_size[0]//32)*(img_size[1]//32)*dims[-1], self.model.config.hidden_size)
        self.image_layer = Linear(1, self.model.config.hidden_size)
        
        self.classifier = AttentionClassifier(query_dim= i_num * v_num, key_dim= t_num * v_num, value_dim= v_num * i_num, num_heads=num_heads, num_class=out_channel, i_num=i_num, t_num=t_num, v_num=v_num)
        
        # ========================= Attention ============================
        self.query_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.key_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.value_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, image, input_text, masked_index):
        # out = self.testmodel(input_text)      
        # # # # # # # # # # # # # # # # # # # # # Text encode # # # # # # # # # # # # # # # # # 
        embeddings = self.text_embeddings(input_text)
        text_encoder_outputs = self.text_encoder(embeddings)
        text_features = text_encoder_outputs[0]
        # pooled_output = self.text_pooler(sequence_output)
        # text_features = pooled_output
        
        # # # # # # # # # # # # # # # # # # # # # Image encode # # # # # # # # # # # # # # # # #
        image_features = self.image_encoder(image)
        num_patches = image_features.size()[1] * image_features.size()[2] * image_features.size()[3]
        image_features = image_features.view(image_features.size()[0], num_patches, -1) 
        image_features = self.image_layer(image_features)
        
        # # # # # # # # # # # # # # # # # # # # # Attention # # # # # # # # # # # # # # # # #
        query = self.query_proj(text_features)  # size [batch_size, word_num, 768]
        key = self.key_proj(image_features)     # size [batch_size, num_patches, 768]
        value = self.value_proj(image_features) # size [batch_size, num_patches, 768]

        # calculate scaled dot-product attention
        d_k = query.size(-1)  # query shape
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        outputs = torch.matmul(attention, value)

        # reshape
        outputs = outputs.reshape(image_features.size()[0], text_features.size()[1], -1)  # 尺寸为[batch_size, word_num, num_patches * 768]
        outputs = self.query_proj(outputs)
        
        # # # # # # # # # # # # # # # # # # # # # Finally predict # # # # # # # # # # # # # # # # # # # # # # # #
        predictions = outputs 
        # 1. instrument text
        instrument = self.instrument_layer(predictions)
        # 2. target text
        target = self.target_layer(predictions)
        # 3. verb text
        verb = self.verb_layer(predictions)
           
        # # predict：Take the predicted index value corresponding to the [MASK] character from the output result。
        predicted_instrument = torch.argmax(instrument[0, masked_index[0]]).item()
        instrument_token = self.instrument_list[predicted_instrument]
        print('Instrument Predicte:', instrument_token)
        
        predicted_target = torch.argmax(target[0, masked_index[1]]).item()
        target_token = self.target_list[predicted_target]
        print('Target Predicte:', target_token)
        
        predicted_verb = torch.argmax(verb[0, masked_index[2]]).item()
        verb_token = self.verb_list[predicted_verb]
        print('Verb Predicte:', verb_token)
        
        # ======================== text output =================================
        text_instrument, text_target, text_verb = instrument[0: ,masked_index[0],:], target[0: ,masked_index[1],:], verb[0: ,masked_index[2],:]
        
        # ======================= image output =================================
        # image output needs 4 outputs, consider transfer text output to image output.
        # max pooling needs to consider the significant words in the text, but I still not sure about this aspect.
        # [batch size, 6] instrument
        image_instrument, _ = torch.max(instrument, dim=1)
        # [batch size, 15] target
        image_target, _ = torch.max(target, dim=1)
        # [batch size, 10] verb
        image_verb, _ = torch.max(verb, dim=1)
        # [batch size, 100] triplet
        # triplet considers the combination of three key words, and needs a special module to fix this output. 
        
        triplet = self.classifier((image_instrument, image_target, image_verb))
        
        # return text_token
        return (text_instrument, text_target, text_verb), (image_instrument, image_target, image_verb, triplet)
    
def weight(model, x):
    image, input_text, masked_index = x
    for i in range(0, 3):
        _ = model(image, input_text, masked_index)
    start_time = time.time()
    _ = model(image, input_text, masked_index)
    end_time = time.time()
    need_time = end_time - start_time
    throughout = round(image.shape[0] / (need_time / 1), 3)
    print('throughout: {} FPS'.format(throughout))


    
if __name__ == '__main__':
    device = 'cuda:0'
    instrument = 'This instrument plays a vital role in the surgical process, helping doctors to operate more precisely, ensuring that each step meets the expected results while reducing errors in operation.'
    target = 'This target is the area that doctors pay close attention to during the operation, and usually needs to be treated with sophisticated technical means to ensure the safety of the operation.'
    verb = 'During surgery, doctors often use this action to precisely control the details of the target area to ensure a smooth operation.'
    
    input_text = '[CLS]' + instrument + '[SEP]' + '[CLS]' + target + '[SEP]' + '[CLS]' + verb + '[SEP]'
    # img_path = "/root/.cache/huggingface/forget/datasets/CholecT45/data/VID01/000000.png"
    # raw_image = Image.open(img_path).convert('RGB')
    # image = torch.tensor(raw_image)
    image = torch.randn(1, 3, 256, 448).to(device)
    
    # load text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased")
    input_text = tokenizer.tokenize(input_text)
    
    # mask: 每句掩码一个词，一共三个
    masked_index = []
    cls_num = 0
    for word_num in range(0, len(input_text)):
        if cls_num == 1 and input_text[word_num] == 'instrument':
            input_text[word_num] = '[MASK]'
            masked_index.append(word_num)
            
        if cls_num == 2 and input_text[word_num] == 'target': 
            input_text[word_num] = '[MASK]'
            masked_index.append(word_num)
        
        if cls_num == 3 and input_text[word_num] == 'action':
            input_text[word_num] = '[MASK]'
            masked_index.append(word_num)
        
        if input_text[word_num] == '[CLS]':
            cls_num += 1
    
    # tensor
    indexed_tokens = tokenizer.convert_tokens_to_ids(input_text)
    input_text = torch.tensor([indexed_tokens]).to(device)
    
    # model
    model = PA(tokenizer).to(device)
    outputs = model(image, input_text, masked_index)
    # weight(model, (image, input_text, masked_index))
    
    
    # ======================================= Train example =====================================
    # TODO: this example only shows how to calculate loss, and needs the user to complete the whole training work.
    model = PA(tokenizer).to(device)
    (text_instrument, text_target, text_verb), (image_instrument, image_target, image_verb, triplet) = model(image, input_text, masked_index)
    
    # ======================================    Triplet     ===================================== 
    tool_weight     = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
    verb_weight     = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448, 2.69230769, 0.60062402]
    target_weight   = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034, 0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]

    loss_functions = {
        'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(device)),
        'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(device)),
        'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(device)),
        'loss_fn_ivt': nn.BCEWithLogitsLoss(),
    }
    
    # target: load target is 0-1 tensor
    instrument_y = torch.zeros(1, 6).to(device)  # instrument
    instrument_y[0][2] = 1
    verb_y = torch.zeros(1, 10).to(device)  # verb
    verb_y[0][2] = 1
    target_y = torch.zeros(1, 15).to(device)  # target
    target_y[0][2] = 1
    triplet_y = torch.zeros(1, 100).to(device) # triplet
    triplet_y[0][2] = 1
    
    image_loss = loss_functions['loss_fn_i'](image_instrument, instrument_y) + loss_functions['loss_fn_v'](image_verb, verb_y) + loss_functions['loss_fn_t'](image_target, target_y) + loss_functions['loss_fn_ivt'](triplet, triplet_y)
    print(image_loss)
    
    # ======================================    Text     =====================================
    loss = nn.CrossEntropyLoss()
    text_loss = loss(text_instrument, instrument_y) + loss(text_target, target_y) + loss(text_verb, verb_y)
    print(text_loss)
    
    
    
    
    