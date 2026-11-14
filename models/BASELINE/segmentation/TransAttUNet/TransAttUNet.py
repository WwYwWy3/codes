"""
@ARTICLE{10244199,
  author={Chen, Bingzhi and Liu, Yishu and Zhang, Zheng and Lu, Guangming and Kong, Adams Wai Kin},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation}, 
  year={2024},
  volume={8},
  number={1},
  pages={55-68},
  keywords={Transformers;Image segmentation;Semantics;Medical diagnostic imaging;Decoding;Convolutional neural networks;Computer architecture;Biomedical image processing;Medical image segmentation;transformer;multi-level guided attention;multi-scale skip connection},
  doi={10.1109/TETCI.2023.3309626}}
"""
# copied from https://github.com/YishuLiu/TransAttUnet

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *
from .unet_parts_att_transformer import *
from ..common import to_three_channels
from .unet_parts_att_multiscale import *

class UNet_Attention_Transformer_Multiscale(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, bilinear=True):
        super(UNet_Attention_Transformer_Multiscale, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(1024, 256 // factor, bilinear)
        self.up3 = Up(512, 128 // factor, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.outc = OutConv(128, n_classes)

        '''位置编码'''
        self.pos = PositionEmbeddingLearned(512 // factor)

        '''空间注意力机制'''
        self.pam = PAM_Module(512)

        '''自注意力机制'''
        self.sdpa = ScaledDotProductAttention(512)

        '''残差多尺度连接'''
        self.fuse1 = MultiConv(768, 256)
        self.fuse2 = MultiConv(384, 128)
        self.fuse3 = MultiConv(192, 64)
        self.fuse4 = MultiConv(128, 64)

    def forward(self, x):
        x = to_three_channels(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        '''Setting 1'''
        x5_pam = self.pam(x5)

        '''Setting 2'''
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos


        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam
        

        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9 = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9)
        return logits

def transattunet(in_ch = 3, class_nums = 1):
    return UNet_Attention_Transformer_Multiscale(n_channels = in_ch, n_classes = class_nums)
