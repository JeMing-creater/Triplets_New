import torch
import timm
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

OUT_HEIGHT = 8
OUT_WIDTH  = 14
# output is a (1, num_features) shaped tensor

class WSL(nn.Module):
    def __init__(self, num_class, in_channels, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=depth, kernel_size=3, padding=1)
        self.cam   = nn.Conv2d(in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn    = nn.BatchNorm2d(depth)
        self.gmp   = nn.AdaptiveMaxPool2d((1,1))
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam     = self.cam(feature)
        logits  = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits

class CAGAM(nn.Module):    
    def __init__(self, num_tool, num_verb, num_target, in_channel=256):
        super(CAGAM, self).__init__()        
        out_depth               = num_tool
        
        #分别生成动作和动作-工具的键值对的对应卷积操作
        self.verb_context       = nn.Conv2d(in_channels=in_channel, out_channels=out_depth, kernel_size=3, padding=1)        
        self.verb_query         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_key           = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key      = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_cmap          = nn.Conv2d(in_channels=out_depth, out_channels=num_verb, kernel_size=1)

        # 分别生成目标和目标-工具的键值对的对应卷积操作
        self.target_context     = nn.Conv2d(in_channels=in_channel, out_channels=out_depth, kernel_size=3, padding=1)     
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


class IVT_Encoder(nn.Module):
    def __init__(self, dim=1, d_state = 16, d_conv = 4, expand = 2, num_slices=4, num_feature=256, num_triplet=100):
        super(IVT_Encoder, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices = num_slices
        )
        self.classifiy = nn.Sequential(
                nn.Linear(num_feature, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, num_triplet)
            )
        
    def forward(self, x):
        x = x.unsqueeze(1).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).squeeze(1)
        x = self.classifiy(x)
        return x


class CrossAttentionModule(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_class=100, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, num_class)  # 输出维度为100
    
    def forward(self, query, key, value):
        B, C_q = query.shape
        B, C_k = key.shape
        B, C_v = value.shape
        
        # Project queries, keys, and values
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        K = self.k_proj(key).view(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        V = self.v_proj(value).view(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, 1)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, num_heads, 1)
        
        # Apply attention to values
        attended_values = torch.matmul(attn_probs, V)  # (B, num_heads, head_dim)
        attended_values = attended_values.view(B, -1)  # (B, query_dim)
        
        # Final projection
        output = self.out_proj(attended_values)  # (B, 100)
        
        return output

class Projection(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128, d_state = 16, d_conv = 4, expand = 2, num_slices=4, drop_rate=0.3):
        super(Projection, self).__init__()
        self.elu    = nn.ELU()  
        self.gap    = nn.AdaptiveAvgPool2d((1,1))
        self.dropout   = nn.Dropout(p=drop_rate)
        self.i_key     = nn.Linear(in_features=num_tool, out_features=out_depth)
        self.v_key     = nn.Linear(in_features=num_verb, out_features=out_depth)
        self.t_key     = nn.Linear(in_features=num_target, out_features=out_depth)
        self.bn1       = nn.BatchNorm1d(out_depth)
        self.bn2       = nn.BatchNorm1d(out_depth)
        self.bn3       = nn.BatchNorm1d(out_depth)
        self.bn4       = nn.BatchNorm1d(num_triplet)
        self.cross_attention = CrossAttentionModule(query_dim=out_depth, key_value_dim=out_depth, num_class=num_triplet)
        self.mamba = Mamba(
                d_model=1, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices = num_slices
        )
        self.shape = nn.Linear(num_triplet*2, num_triplet)
        
    def forward(self, cam_i, cam_v, cam_t, X):  
        cam_i = self.gap(cam_i).squeeze(-1).squeeze(-1)
        cam_v = self.gap(cam_v).squeeze(-1).squeeze(-1)
        cam_t = self.gap(cam_t).squeeze(-1).squeeze(-1)
        
        q = self.elu(self.bn1(self.i_key(cam_i)))
        k = self.elu(self.bn2(self.v_key(cam_v)))
        v = self.elu(self.bn3(self.t_key(cam_t)))
        
        cam_X = self.cross_attention(q, k, v)
        X = self.elu(self.bn4(self.dropout(X)))
        
        feature_X = torch.cat((cam_X, X), dim=1)
        
        feature_X = self.mamba(feature_X.unsqueeze(1).transpose(-1, -2))
        feature_X = feature_X.transpose(-1, -2).squeeze(1)
        X = self.shape(feature_X)
        return X

class Decoder(nn.Module):
    def __init__(self, layer_size=8, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128, d_state = 16, d_conv = 4, expand = 2, num_slices=4, drop_rate=0.3):
        super(Decoder, self).__init__()
        self.projection = nn.ModuleList([Projection(num_tool=num_tool, num_verb=num_verb, num_target=num_target, num_triplet=num_triplet, out_depth=out_depth, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices, drop_rate=drop_rate) for i in range(layer_size)])
        self.classifiy = nn.Linear(num_triplet, num_triplet)
        
    def forward(self, enc_i, enc_v, enc_t, enc_ivt):   
        for P in zip(self.projection): 
            x = P[0](enc_i, enc_v, enc_t, enc_ivt)
            enc_ivt = enc_ivt + x
        x = self.classifiy(enc_ivt)
        return x

class Encoder(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, d_state = 16, d_conv = 4, expand = 2, num_slices=64, depth=64):
        super(Encoder, self).__init__()
        self.basemodel = timm.create_model(
            'samvit_base_patch16.sa1b',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.num_feature = self.basemodel.num_features
        self.wsl         = WSL(num_tool, self.num_feature, depth)
        self.cagam       = CAGAM(num_tool, num_verb, num_target, self.num_feature)
        self.bottleneck  = IVT_Encoder(dim=1, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices, num_feature=self.num_feature, num_triplet=num_triplet)
        
    def forward(self, x):
        high_x = self.basemodel.forward_features(x)
        low_x = self.basemodel(x)
        enc_i = self.wsl(high_x)
        enc_v, enc_t  = self.cagam(high_x, enc_i[0])
        enc_ivt = self.bottleneck(low_x)
        return enc_i, enc_v, enc_t, enc_ivt

class SAMMba(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, d_state = 16, d_conv = 4, expand = 2, num_slices=8, depth=64, out_depth=128, drop_rate=0.3):
        super(SAMMba, self).__init__()
        self.encoder = Encoder(num_tool=num_tool, num_verb=num_verb, num_target=num_target, num_triplet=num_triplet, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices, depth=depth)
        self.decoder = Decoder(layer_size=layer_size, num_tool=num_tool, num_verb=num_verb, num_target=num_target, num_triplet=num_triplet, out_depth=out_depth, d_state = d_state, d_conv = d_conv, expand = expand, num_slices=num_slices, drop_rate=drop_rate)
        
    def forward(self, image): 
        enc_i, enc_v, enc_t, enc_ivt = self.encoder(image)
        enc_ivt = self.decoder(enc_i[0], enc_v[0], enc_t[0], enc_ivt)
        return enc_i, enc_v, enc_t, enc_ivt
    
if __name__ == '__main__':
    device = 'cuda:0'
    image = torch.randn(20, 3, 256, 448).to(device)
    
    model = SAMMba().to(device)
    
    enc_i, enc_v, enc_t, enc_ivt = model(image)
    print(enc_i[1].shape)
    print(enc_v[1].shape)
    print(enc_t[1].shape)
    print(enc_ivt.shape)
