import torch
import timm
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model = 1, d_state = 16, d_conv = 4, expand = 2, num_slices=64):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices = num_slices
        )
        
    def forward(self, x):  
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        return out

class MambaAttention(nn.Module):
    def __init__(self, d_model = 1, d_state = 16, d_conv = 4, expand = 2, num_slices=64):
        super(MambaAttention, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices = num_slices
        )
        self.bn = nn.BatchNorm2d(d_model)
        # self.avg  = nn.Conv2d(in_channels=196, out_channels=d_model, kernel_size=3, padding=1) 
        
    def get_feature_m(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm).transpose(-1, -2)
        
        return x_mamba
    
    def forward(self, x1, x2):
        # b, c = x1.shape[:2] 
        batch_size, channel = x1.shape[:2] 
        img_dims = x1.shape[2:]
        A = self.get_feature_m(x1)
        B = self.get_feature_m(x2)
        dot_product = torch.matmul(A, B.transpose(-2, -1))
        result = torch.matmul(dot_product, B)
        result = result.transpose(-1, -2).reshape(batch_size, channel, *img_dims)
        result = self.bn(result)
        return result

class BaseModel(nn.Module):
    def __init__(self, in_channel = 3, class_num=6, d_state = 16, d_conv = 4, expand = 2, drop_rate=0.3, dim = [16, 32, 64, 128], num_slices = [64, 32, 16, 4], block_depth = [2, 2, 2, 2], need_class=False):
        super(BaseModel, self).__init__()
        self.down1 = nn.Conv2d(in_channel, dim[0], kernel_size=3, stride=2, padding=1)
        block = []
        for _ in range(block_depth[0]):
            block.append(MambaBlock(d_model = dim[0], d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices[0]))
        self.block1 = nn.Sequential(*block)
        self.down11 = nn.Conv2d(in_channel, dim[0], kernel_size=3, stride=2, padding=1)
        
        self.down2 = nn.Conv2d(dim[0]*2, dim[1], kernel_size=3, stride=2, padding=1)
        block = []
        for _ in range(block_depth[1]):
            block.append(MambaBlock(d_model = dim[1], d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices[1]))
        self.block2 = nn.Sequential(*block)
        self.down22 = nn.Conv2d(dim[0]*2, dim[1], kernel_size=3, stride=2, padding=1)
        
        self.down3 = nn.Conv2d(dim[1]*2, dim[2], kernel_size=3, stride=2, padding=1)
        block = []
        for _ in range(block_depth[2]):
            block.append(MambaBlock(d_model = dim[2], d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices[2]))
        self.block3 = nn.Sequential(*block)
        self.down33 = nn.Conv2d(dim[1]*2, dim[2], kernel_size=3, stride=2, padding=1)
        
        self.down4 = nn.Conv2d(dim[2]*2, dim[3], kernel_size=3, stride=2, padding=1)
        block = []
        for _ in range(block_depth[3]):
            block.append(MambaBlock(d_model = dim[3], d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices[3]))
        self.block4 = nn.Sequential(*block)
        # self.down44 = nn.Conv2d(dim[2]*2, dim[3], kernel_size=3, stride=2, padding=1)
        self.dropout   = nn.Dropout(p=drop_rate)
        
        self.need_class =need_class
        if need_class == True:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifiy = nn.Sequential(
                    nn.Linear(dim[3], dim[3]*2),
                    nn.ELU(),
                    nn.Linear(dim[3]*2, class_num)
                )
    
    def forward(self, x): 
        x1 = self.down1(x)
        x1 = self.block1(x1)
        
        x11 = self.down11(x)
        
        x = torch.cat([x1, x11], dim=1)
        
        x2 = self.down2(x)
        x2 = self.block2(x2)
        
        x22 = self.down22(x)
        
        x = torch.cat([x2, x22], dim=1)
        x3 = self.down3(x)
        x3 = self.block3(x3)
        
        x33 = self.down33(x)
        
        x = torch.cat([x3, x33], dim=1)
        x4 = self.down4(x)
        x4 = self.block4(x4)
        
        x4 = self.dropout(x4)
        
        if self.need_class==True:
            x = self.global_avg_pool(x4)
            x = x.view(x.size(0), -1)
            x = self.classifiy(x)
            return x, x4
        else:
            return x4

class CAGAM(nn.Module):    
    def __init__(self, dim, verb_num=6, target_num=10, d_state = 16, d_conv = 4, expand = 2, num_slices= 4):
        super(CAGAM, self).__init__()  
        self.mamba1 = MambaAttention(d_model = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices)
        self.mamba2 = MambaAttention(d_model = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.verb_classifiy = nn.Sequential(
                nn.Linear(dim, dim*2),
                nn.ELU(),
                nn.Linear(dim*2, verb_num)
            )
        self.target_classifiy = nn.Sequential(
                nn.Linear(dim, dim*2),
                nn.ELU(),
                nn.Linear(dim*2, target_num)
            )
        
    def get_verb(self, raw, cam):
        out = self.mamba1(raw, cam)
        x = self.global_avg_pool(out)
        x = x.view(x.size(0), -1)
        x = self.verb_classifiy(x)
        return x, out
    
    def get_target(self, raw, cam):
        out = self.mamba2(raw, cam)
        x = self.global_avg_pool(out)
        x = x.view(x.size(0), -1)
        x = self.target_classifiy(x)
        return x, out
    
    def forward(self, verb_feature, target_feature, tool_feature):
        verb, verb_feature = self.get_verb(verb_feature, tool_feature)
        target, target_feature = self.get_target(target_feature, tool_feature)
        return (verb, verb_feature), (target, target_feature)

class TriEncoder(nn.Module):
    def __init__(self, in_channel = 3, num_tool=6, num_verb=10, num_target=15, d_state = 16, d_conv = 4, expand = 2, drop_rate=0.3, dim = [16, 32, 64, 128], num_slices = [64, 32, 16, 4], block_depth = [2, 2, 2, 2]):
        super(TriEncoder, self).__init__()
        self.base_model_1 = BaseModel(in_channel = in_channel, class_num=num_tool, d_state = d_state, d_conv = d_conv, expand = expand, dim = dim, drop_rate=drop_rate, num_slices = num_slices, block_depth = block_depth, need_class=True)
        self.base_model_2 = BaseModel(in_channel = in_channel, class_num=num_verb, d_state = d_state, d_conv = d_conv, expand = expand, dim = dim, drop_rate=drop_rate, num_slices = num_slices, block_depth = block_depth, need_class=False)
        self.base_model_3 = BaseModel(in_channel = in_channel, class_num=num_target, d_state = d_state, d_conv = d_conv, expand = expand, dim = dim, drop_rate=drop_rate, num_slices = num_slices, block_depth = block_depth, need_class=False)
        self.cagam        = CAGAM(dim[-1], verb_num=num_verb, target_num=num_target, d_state = d_state, d_conv = d_conv, expand = expand, num_slices= num_slices[-1])
        
    def forward(self, image): 
        tool, tool_feature  = self.base_model_1(image)
        verb_feature        = self.base_model_2(image)
        target_feature      = self.base_model_3(image)
        (verb, verb_feature), (target, target_feature) = self.cagam(verb_feature, target_feature, tool_feature)
        return (tool, tool_feature), (verb, verb_feature), (target, target_feature)

class Dobule(nn.Module):
    def __init__(self, dim = 128, d_state = 16, d_conv = 4, expand = 2, num_slices = 4):
        super(Dobule, self).__init__()
        # self.avg = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0)
        # self.mamba = MambaBlock(d_model = dim*2, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices)
        self.mamba = MambaAttention(d_model = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices)
        
    def forward(self, first_feature, second_feature): 
        # feature = torch.cat([first_feature, second_feature], dim=1) 
        feature = self.mamba(first_feature, second_feature)  
        # feature = self.avg(feature)
        return feature

class InteractionBlock(nn.Module):
    def __init__(self, dim = 128, d_state = 16, d_conv = 4, expand = 2, num_slices = 4):
        super(InteractionBlock, self).__init__()
        self.tv = Dobule(dim = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices = num_slices)
        self.tt = Dobule(dim = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices = num_slices)
        self.vt = Dobule(dim = dim, d_state = d_state, d_conv = d_conv, expand = expand, num_slices = num_slices)
        self.mamba = MambaAttention(d_model = dim*3, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices)

    def forward(self, input): 
        tool_feature, verb_feature, target_feature, triplet = input
        tv = self.tv(tool_feature, verb_feature)
        tt = self.tt(tool_feature, target_feature)
        vt = self.vt(verb_feature, target_feature)
        o_triplet = triplet
        triplet = torch.cat([tv, tt, vt], dim=1)
        triplet = self.mamba(triplet, o_triplet)
        return (tool_feature, verb_feature, target_feature, triplet)
             
class TriDecoder(nn.Module):
    def __init__(self, d_model=128, num_triplet=100, block_layer = 8, d_state = 16, d_conv = 4, expand = 2, drop_rate=0.3, num_slices = 4):
        super(TriDecoder, self).__init__()
        block = []
        for _ in range(block_layer):
            block.append(InteractionBlock(dim = d_model, d_state = d_state, d_conv = d_conv, expand = expand, num_slices = num_slices))
        self.block = nn.Sequential(*block)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifiy = nn.Sequential(
                nn.Linear(d_model*3, d_model),
                nn.ELU(),
                nn.Linear(d_model, num_triplet)
            )
        
    def forward(self, tool_feature, verb_feature, target_feature): 
        triplet = torch.cat([tool_feature, verb_feature, target_feature], dim=1)
        _, _, _, triplet = self.block((tool_feature, verb_feature, target_feature, triplet))
        triplet = self.global_avg_pool(triplet).view(triplet.size(0), -1)
        triplet = self.classifiy(triplet)
        return triplet
        
class TriBase(nn.Module):
    def __init__(self, in_channel = 3, num_tool=6, num_verb=10, num_target=15, num_triplet=100, d_state = 16, d_conv = 4, expand = 2, drop_rate=0.3, dim = [16, 32, 64, 128], num_slices = [64, 32, 16, 4], block_depth = [2, 2, 2, 2], block_layer = 8):
        super(TriBase, self).__init__()
        self.encoder = TriEncoder(in_channel = in_channel, num_tool=num_tool, num_verb=num_verb, num_target=num_target, d_state = d_state, d_conv = d_conv, expand = expand, drop_rate=drop_rate, dim = dim, num_slices = num_slices, block_depth = block_depth)
        self.decoder = TriDecoder(d_model=dim[-1], num_triplet=num_triplet, block_layer = block_layer, d_state = d_state, d_conv = d_conv, expand = expand, drop_rate=0.3, num_slices = num_slices[-1])
    
    def forward(self, image): 
        (tool, tool_feature), (verb, verb_feature), (target, target_feature) = self.encoder(image)
        triplet = self.decoder(tool_feature, verb_feature, target_feature)
        output  = torch.cat([triplet, tool, verb, target], dim=1)
        return output

if __name__ == '__main__':
    device = 'cuda:0'
    image = torch.randn(20, 3, 224, 224).to(device)
    model = TriBase().to(device)
    
    out = model(image)
    print(out.shape)
    