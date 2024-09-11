import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from timm.models.layers.helpers import to_2tuple
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from .conv_fea import *

drop = 0.1
num_layer = 4
class Mlp(nn.Module):
    def __init__(self, inchanels, outchanels):
        super().__init__()
        self.clr_fc = nn.Sequential(
                                    nn.Linear(inchanels, outchanels * 2),
                                    nn.Dropout(drop),
                                    nn.GELU(),
                                    nn.Linear(outchanels * 2, outchanels),
                                    nn.Dropout(drop)
                                    )
    def forward(self, x):
        return self.clr_fc(x)

class RD_Net(nn.Module):
    def __init__(self, channels): 
        super(RD_Net, self).__init__()
        self.conv1_3 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(1, channels, 3, 1, 0),
                                    nn.BatchNorm2d(channels),
                                    nn.LeakyReLU()
                                    )
        self.conv2_1 = nn.Conv2d(channels, channels * 4, 1, 1, 0)
        self.Dense_R_Block1 = Dense_Res_block(channels = channels, outchannels = channels * 2)
        self.Dense_R_Block2 = Dense_Res_block(channels = channels * 2, outchannels = channels * 4)
     
        self.norm_64 = nn.LayerNorm(channels * 4)
   
        self.MLP_64 = Mlp(channels * 4, channels * 4)
        self.Depwise_con64 = nn.Conv2d(channels * 4,channels * 4,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              groups=channels * 4,
                              bias=False)
  

    def forward(self, x):
        x = self.conv1_3(x)
        fea_1 = self.Dense_R_Block1(x)
        fea_1 = self.Dense_R_Block2(fea_1)
        b, c, h, w = fea_1.shape # 16, 64, 128, 128  
        fea_out = fea_1.permute(0, 2, 3, 1)    # 16 128 128 64
        
        for i in range(num_layer):
            fea_res = fea_out
            fea_norm = self.norm_64(fea_out) 
            fea_norm = rearrange(fea_norm, 'b h w c -> b (h w) c') 
            fea_out = self.MLP_64(fea_norm)  
            fea_out = rearrange(fea_out, 'b (h w) c  ->b h w c ', h = h, w = w, c = c)
            fea_out = fea_res + self.norm_64(fea_out)
            fea_out = self.Depwise_con64(fea_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)         
        fea_out = fea_1 + fea_out.permute(0, 3, 1, 2) 
 
        fea_out = self.conv2_1(x) +fea_out
        return fea_out, b, c, h, w


class ConvMLP(nn.Module):
    def __init__(self, channel):
        super(ConvMLP,self).__init__()
        self.norm_128 = nn.LayerNorm(channel * 2)
        self.pre = RD_Net(channel // 4 )
        self.clr_fc = nn.Sequential(
                                    nn.Linear(channel * 2, channel * 4),  
                                    nn.Dropout(drop),
                                    nn.GELU(),
                                    nn.Linear(channel * 4, channel * 2),
                                    nn.Dropout(drop)
                                    )
        self.Depwise_con128 = nn.Conv2d(channel * 2,channel * 2,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              groups=channel * 2,
                              bias=False)

    def forward(self, ir, vis):

        fea_ir, _, _, _, _ = self.pre.forward(ir)
        fea_vis, _, _, _, _ = self.pre.forward(vis)
        fea_fusion1 = torch.cat((fea_ir,fea_vis), 1)
        
        b, c, h, w = fea_fusion1.shape  # 16, 128, 128, 128
        fea_fusion = fea_fusion1.permute(0, 2, 3, 1)  # 16 128 128 128(channel)
   
        for i in range(num_layer):
            fea_res = fea_fusion
            fea_norm = self.norm_128(fea_fusion) 
            fea_norm = rearrange(fea_norm, 'b h w c -> b (h w) c') 
            fea_out = self.clr_fc(fea_norm)  
            fea_out = rearrange(fea_out, 'b (h w) c  ->b h w c ', h = h, w = w, c = c)
            fea_out = fea_res + self.norm_128(fea_out)
            fea_out = self.Depwise_con128(fea_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)   
        fea_fusion = fea_fusion1 + fea_out.permute(0, 3, 1, 2) 
        
        return fea_fusion, fea_ir, fea_vis

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv_mlp = ConvMLP(64)
        self.reconstruction = Feature_reconstruction()

    def forward(self, ir, vis):
        fea, fea_ir, fea_vis = self.conv_mlp(ir, vis)
        fusion_image = self.reconstruction(fea)

        return fusion_image, fea_ir, fea_vis

if __name__ == "__main__":
    x = torch.tensor(np.random.rand(16, 1, 128, 128).astype(np.float32))
    model = FusionNet()
    y = model(x, x)
    print('test ok!')
    
