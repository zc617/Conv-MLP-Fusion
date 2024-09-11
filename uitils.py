import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutil
import cv2
# import pytorch_msssim
from loss_ssim import *
import torchvision as tv


def gradient(input):


    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).to(input.device)
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).to(input.device)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient

def clamp(value, min=0., max=1.0):

    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
 
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
 
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out

def Int_Loss(fused_image, vis_image, inf_image, w1_vis):

    loss_Int: object = F.l1_loss(fused_image,inf_image) + w1_vis * F.l1_loss(fused_image, vis_image)
    return loss_Int


def gradinet_Loss(fused_image, vis_image, inf_image):
    gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_image)))

    return gradinet_loss


