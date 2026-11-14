# copied from https://github.com/wmy3003/FBAMPSM/tree/main

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .pvtv2 import pvt_v2_b2
from .FBHA import FBHA
from .MPSM import MPSM
from ...common import to_three_channels

import warnings

warnings.filterwarnings("ignore")

def fbampsm(in_channels=3, out_channels = 1):
    return FBAMPSM(in_channels, out_channels)

class FBAMPSM(nn.Module):
    def __init__(self, in_channels=3, out_channels = 1):
        super().__init__()
        self.decoder1 = MPSM(64)
        self.decoder2 = MPSM(128)
        self.decoder3 = MPSM(320)
        self.decoder4 = MPSM(512)

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
        save_model = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.attention1= FBHA(64, 3, True,True)
        self.attention2 = FBHA(128, 2, True,True)
        self.attention3 = FBHA(320, 1, True,True)
        self.attention4 = FBHA(512, 0, True, True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 320, kernel_size=1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """Encoder"""
        x = to_three_channels(x)
        H, W = x.size()[2:]
        x1, x2, x3, x4 = self.backbone(x)
        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4

        attention_x1 = self.attention1(x1)
        attention_x2 = self.attention2(x2)
        attention_x3 = self.attention3(x3)
        attention_x4 = self.attention4(x4)

        skip4 = skip4 + attention_x4
        x3_ = self.decoder4(skip4)
        x3_ = self.conv4(x3_)
        x3_ = F.interpolate(x3_, size=(H//16, W//16), mode="bilinear", align_corners=False)

        skip3 = skip3 + attention_x3
        x3_ = x3_ + skip3
        x2_ = self.decoder3(x3_)
        x2_ = self.conv3(x2_)
        x2_ = F.interpolate(x2_, size=(H//8, W//8), mode="bilinear", align_corners=False)

        skip2 = skip2 + attention_x2
        x2_ = x2_ + skip2
        x1_ = self.decoder2(x2_)
        x1_ = self.conv2(x1_)
        x1_ = F.interpolate(x1_, size=(H//4, W//4), mode="bilinear", align_corners=False)

        skip1 = skip1 + attention_x1
        x1_ = x1_ + skip1
        x_ = self.decoder1(x1_)
        x_ = self.conv1(x_)
        x_ = F.interpolate(x_, size=(H, W), mode="bilinear", align_corners=False)

        x_ = self.final_conv(x_)
        
        return x_

if __name__ == '__main__':
    model = FBAMPSM(3, 1).to('cuda')
    x = torch.randn(4, 3, 224, 224).to('cuda')
    y = model(x)
    print(y.shape)
