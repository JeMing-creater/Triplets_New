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

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from transformers import GPT2Tokenizer,GPT2LMHeadModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

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


class DynamicAttention(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(DynamicAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, target_dim)

    def forward(self, x, target_length):
        _, _, feature_dim = x.size()
        assert feature_dim == self.query.out_features

        # 创建query, key, value
        Q = self.query(x)  # [batch_size, seq_length, feature_dim]
        K = self.key(x)    # [batch_size, seq_length, feature_dim]
        V = self.value(x)  # [batch_size, seq_length, target_dim]

        # 自注意力机制的核心操作
        scores = torch.bmm(Q, K.transpose(1, 2))
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)

        # 根据目标长度调整输出尺寸
        output = context[:, :target_length, :]
        return output



class PA(nn.Module):
    def __init__(self, tokenizer, in_channel = 3, out_channel = 100, i_num = 6, t_num=15, v_num=10, num_heads=10, dims = [64, 128, 320, 512], model_dir='/root/.cache/huggingface/forget/pvt_v2_b3.pth'):
        super(PA, self).__init__()  
        
        self.instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
        self.target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'null_target']       
        self.verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'null_verb']      
      
        # ======================== text ===============================
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        
        # 修改 embedding 维度
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # ======================== image ===============================
        self.image_encoder = pvt_v2_b2(in_chans=in_channel, embed_dims=dims)
        if os.path.isfile(model_dir):
            save_model = torch.load(model_dir)
            model_dict = self.image_encoder.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.image_encoder.load_state_dict(model_dict)
            
        self.MultimodalConnector = nn.Linear(1, self.model.bert.config.hidden_size)
        self.AttentionPool = DynamicAttention(self.model.bert.config.hidden_size, self.model.bert.config.hidden_size)
    
    def get_instrument(self):
        word_indices = self.tokenizer.convert_tokens_to_ids(self.instrument_list)
        selected_confidences = self.text_instrument[:, word_indices]
        return selected_confidences
    
    def get_target(self):
        word_indices = self.tokenizer.convert_tokens_to_ids(self.target_list)
        selected_confidences = self.text_target[:, word_indices]
        return selected_confidences
    
    def get_verb(self):
        word_indices = self.tokenizer.convert_tokens_to_ids(self.verb_list)
        selected_confidences = self.text_verb[:, word_indices]
        return selected_confidences
    
    def print_predict(self):
        instrument = torch.argmax(self.text_instrument).item()
        print('Instrument Predicte:', self.instrument_list[instrument])
        target = torch.argmax(self.text_target).item()
        print('Target Predicte:', self.target_list[target])
        verb = torch.argmax(self.text_verb).item()
        print('Verb Predicte:', self.verb_list[verb])
        
    def forward(self, input_image, input_text, masked_index):
        
        # text embeddings
        text_embedding = self.model.bert.embeddings(input_text)
        
        # image embeddings
        image_embedding = self.image_encoder(input_image)
        num_patches = image_embedding.size()[1] * image_embedding.size()[2] * image_embedding.size()[3]
        image_embedding = image_embedding.view(image_embedding.size()[0], num_patches, -1) 
        image_embedding = self.MultimodalConnector(image_embedding)
        image_embedding = self.AttentionPool(image_embedding, text_embedding.size()[1])
        
        
        # add and division
        embedding = torch.add(text_embedding, image_embedding) / 2.0
        # embedding = torch.add(text_embedding, image_embedding)
        # ======================= Language model  ================================
        encoder_outputs = self.model.bert.encoder(embedding)
        sequence_output = encoder_outputs[0]
        
        bert_out = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=None,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        
        se_output = bert_out[0]
        prediction_scores = self.model.cls(se_output)
        
        outputs = MaskedLMOutput(
            loss=None,
            logits=prediction_scores,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )
        
        predictions = outputs[0]
        
        self.text_instrument = predictions[0: ,masked_index[0],:]
        self.text_target = predictions[0: ,masked_index[1],:]
        self.text_verb = predictions[0: ,masked_index[2],:]
        
        self.text_instrument = self.get_instrument()
        self.text_target = self.get_target()
        self.text_verb = self.get_verb()
        
        # self.print_predict()
        
        return self.text_instrument, self.text_target, self.text_verb
        

# 修改词典
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
    
if __name__ == '__main__':
    device = 'cuda:1'
    instrument = 'This instrument plays a vital role in the surgical process, helping doctors to operate more precisely, ensuring that each step meets the expected results while reducing errors in operation.'
    target = 'This target is the area that doctors pay close attention to during the operation, and usually needs to be treated with sophisticated technical means to ensure the safety of the operation.'
    verb = 'During surgery, doctors often use this action to precisely control the details of the target area to ensure a smooth operation.'
    
    input_text = '[CLS]' + instrument + '[SEP]' + '[CLS]' + target + '[SEP]' + '[CLS]' + verb + '[SEP]'
    # img_path = "/root/.cache/huggingface/forget/datasets/CholecT45/data/VID01/000000.png"
    # raw_image = Image.open(img_path).convert('RGB')
    # image = torch.tensor(raw_image)
    image = torch.randn(1, 3, 256, 448).to(device)
    
    instrument_list = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator'] 
    target_list = ['gallbladder', 'cystic_plate', 'cystic_duct','cystic_artery', 'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen_bag', 'null_target']       
    verb_list = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'null_verb']      
        
    add_list = instrument_list + target_list + verb_list
    # load textokenizert
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    add_tokens_tokenizer(tokenizer, add_list)
    
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
            
    input_image = torch.randn(1, 3, 256, 448).to(device)
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(input_text)
    input_text = torch.tensor([indexed_tokens]).to(device)
    
    model = PA(tokenizer).to(device)
    model(input_image, input_text, masked_index)
    
    
    
    