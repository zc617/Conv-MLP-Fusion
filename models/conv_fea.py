import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os

class Conv3_Bn_LeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_LeakyRelu2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(self.refpadding(x))))
    
class Conv3_LeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_LeakyRelu2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.lrelu(self.conv(self.refpadding(x)))
    
class Conv3_Bn_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_Relu2d, self).__init__()
        self.ref_padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.ref_padding(x))))
     
class Feature_reconstruction(nn.Module):
    def __init__(self):
        super(Feature_reconstruction, self).__init__()
        self.conv3_1 = Conv3_Bn_Relu2d(128, 128)
        self.conv3_2 = Conv3_Bn_Relu2d(128, 64)
        self.conv3_3 = Conv3_Bn_Relu2d(64, 32)
        self.conv3_4 = Conv3_Bn_Relu2d(32, 16)
        self.conv3_5 = Conv3_Bn_Relu2d(16, 1)
        
    def forward(self,feature):
        conv3_1 = self.conv3_1(feature)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        conv3_5 = self.conv3_5(conv3_4)
        return conv3_5
    
class Dense_Res_block(nn.Module):
    def __init__(self, channels, outchannels):
         super(Dense_Res_block, self).__init__()
         self.conv2 = Conv3_Bn_LeakyRelu2d(channels, channels)
         self.conv3 = Conv3_Bn_LeakyRelu2d(channels * 2, channels) 
         self.conv5 = nn.Conv2d(channels * 3, outchannels, 1, 1, 0)
         self.conv6 = nn.Conv2d(channels, outchannels, 1, 1, 0)
         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
         self.tanh = nn.Tanh()
        #  
    def forward(self, x):
         x1 = self.conv2(x) 
         x2 = self.conv3(torch.cat((x1, x), 1))  
         x3 = self.tanh(self.conv5(torch.cat((x2, torch.cat((x1, x), 1)), 1)))
         x5 = self.lrelu(self.conv6(x))
         return x5 + x3 
