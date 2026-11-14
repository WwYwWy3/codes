import os
import sys
import warnings
from typing import Dict

from .AttUNet import attentionunet
from .CFFormer import cdformer
from .MobileUViT import mobileuvit, mobileuvit_l
from .ResUKAN import resukan
from .SwinUNet import swinunet
from .TransAttUNet import transattunet
from .UKAN import ukan
from .UNeXt import unext
from .UNet import unet
from .UNet3Plus import unet3plus
from .WCENet import wcenet # 2026
from .FBAMPSM.models import fbampsm # 2026
warnings.filterwarnings("ignore")

def create_baseline(model_name, params: Dict):
    if model_name == 'unet':
        return unet(**params)
    elif model_name == 'attentionunet':
        return attentionunet(**params)
    elif model_name == 'cdformer':
        return cdformer(**params)
    elif model_name == 'mobileuvit':
        return mobileuvit(**params)
    elif model_name == 'mobileuvit_l':
        return mobileuvit_l(**params)
    elif model_name == 'resukan':
        return resukan(**params)
    elif model_name == 'swinunet':
        return swinunet(**params)
    elif model_name == 'transattunet':
        return transattunet(**params)
    elif model_name == 'ukan':
        return ukan(**params)
    elif model_name == 'unet3plus':
        return unet3plus(**params)
    elif model_name == 'unext':
        return unext(**params)
    elif model_name == 'wcenet':
        return wcenet(**params)
    elif model_name == 'fbampsm':
        return fbampsm(**params)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def YAML2DICT(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    size = (4, 3, 224, 224)
    
    model_name = ['unet3plus', 'unext', 'wcenet', 'fbampsm']
    save_path = '/mnt/f/PatternRecognition/codes/models/BASELINE/segmentation/model_details.txt'
    configs = YAML2DICT('/mnt/f/PatternRecognition/codes/config/models.yaml')['model_info']
    import torch

    for name in model_name:
        model = create_baseline(name, configs[name]).to('cuda')
        x = torch.randn(size).to('cuda')
        y = model(x)
        
        print(f'\n{name}:')
        print(f'Output shape: {y.shape}')
        print(f'Input size: {x.shape}')
        
        # try:
        #     loss = y.sum()
        #     model.zero_grad()
        #     loss.backward()
            
        #     grad_ok = True
        #     no_grad_params = []
        #     nan_grad_params = []
        #     for pname, p in model.named_parameters():
        #         if p.requires_grad:
        #             if p.grad is None:
        #                 no_grad_params.append(pname)
        #             elif torch.isnan(p.grad).any():
        #                 nan_grad_params.append(pname)
            
        #     if no_grad_params:
        #         print(f'⚠️ 无梯度的参数: {no_grad_params[:5]}...')
        #         grad_ok = False
        #     if nan_grad_params:
        #         print(f'❌ 梯度包含 NaN 的参数: {nan_grad_params[:5]}...')
        #         grad_ok = False
            
        #     if grad_ok:
        #         print('✅ 反向传播成功，梯度正常')
        # except Exception as e:
        #     print(f'❌ 反向传播失败: {e}')
