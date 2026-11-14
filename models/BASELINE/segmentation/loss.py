import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU, SurfaceDistanceMetric
from monai.losses import DiceLoss
import numpy as np
from typing import Dict, List, Optional, Any

def calc_dice(pred: torch.Tensor, gt: torch.Tensor, class_nums: int, have_channel: bool = True, include_background: bool = False, reduction: str = "mean"):
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    gt = transform_multi_to_onehot(gt, class_nums, have_channel)
    dice_metric(y_pred = pred, y = gt)
    result = dice_metric.aggregate()
    return result

def transform_multi_to_onehot(gt: torch.Tensor, class_nums: int, have_channel: bool = False):
    # gt: [batch, 1, ...] 或 [batch, ...]，2D / 3D 通用
    if have_channel:
        gt_ = gt.squeeze(1).to(torch.long)
    else:
        gt_ = gt.to(torch.long)
    gt_onehot = F.one_hot(gt_, num_classes = class_nums).movedim(-1, 1).float() # [batch, class_nums, ...]
    return gt_onehot

class LossFunction(nn.Module):
    def __init__(self, w_dice: float = 0.5, w_ce: float = 1, class_nums: int = 14, gt_have_channel: bool = False) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.class_nums = class_nums
        self.gt_have_ch = gt_have_channel
        # class_nums == 1 走二分类（sigmoid + BCE），否则多分类（softmax + CE）
        # softmax/sigmoid=True 表示 pred 直接传 logits
        if class_nums == 1:
            self.dice_loss = DiceLoss(include_background=True, sigmoid=True, reduction="mean")
        else:
            self.dice_loss = DiceLoss(include_background=False, softmax=True, reduction="mean")

    def forward(self, pred, gt):
        # pred: [batch, class, ...] logits; gt: [batch, ...] 或 [batch, 1, ...] 标签图
        target = gt.squeeze(1) if self.gt_have_ch else gt
        if self.class_nums == 1:
            dice = self.dice_loss(pred, target.float().unsqueeze(1))
            ce = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())
        else:
            dice = self.dice_loss(pred, transform_multi_to_onehot(target, self.class_nums))
            ce = F.cross_entropy(pred, target.long())
        return self.w_dice * dice + self.w_ce * ce

if __name__ == '__main__':
    # 用 BTCV 的真实切片验证 LossFunction
    root = '/mnt/f/PatternRecognition/codes/datasets/BTCV/processed_data'
    class_nums = 14

    image = torch.from_numpy(np.load(f'{root}/image/img0026/75.npz')['img']).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
    image = (image - image.mean()) / (image.std() + 1e-8)  # CT 原始值约 [-3024, 1226]，与训练时一致做标准化
    gt = torch.from_numpy(np.load(f'{root}/mask/label0026/75.npz')['mask']).unsqueeze(0).long()                 # [1, H, W]
    _, _, height, width = image.shape
    print(f'image {tuple(image.shape)}  gt {tuple(gt.shape)}  gt 中出现的类别 {sorted(set(gt.flatten().tolist()))}')

    criterion = LossFunction(w_dice=0.5, w_ce=1, class_nums=class_nums)

    def report(name, pred):
        dice = criterion.dice_loss(pred, transform_multi_to_onehot(gt, class_nums))
        ce = F.cross_entropy(pred, gt)
        loss = criterion(pred, gt)
        print(f'{name:<10} dice_loss={dice.item():.4f}  ce={ce.item():.4f}  loss={loss.item():.4f}')

    # 1) 完美预测：dice_loss 应为 0，ce 应为 0
    onehot_gt = F.one_hot(gt[0], class_nums).movedim(-1, 0).unsqueeze(0).float()
    perfect = torch.where(onehot_gt > 0, torch.tensor(20.0), torch.tensor(-20.0))
    report('perfect', perfect)

    # 2) 全部预测为背景：dice_loss 应为 1
    all_bg = torch.full((1, class_nums, height, width), -20.0)
    all_bg[:, 0] = 20.0
    report('all-bg', all_bg)

    # 3) 随机 logits
    report('random', torch.randn(1, class_nums, height, width))

    # 4) 真实图像过一层未训练卷积
    model = nn.Conv2d(1, class_nums, kernel_size=1)
    report('conv2d', model(image))

    # 5) 只用 dice 项反传，确认 w_dice 现在真的参与训练
    dice_only = LossFunction(w_dice=1, w_ce=0, class_nums=class_nums)
    loss = dice_only(model(image), gt)
    loss.backward()
    print(f'dice-only:  loss={loss.item():.4f}  conv.weight.grad='
          f'{None if model.weight.grad is None else float(model.weight.grad.abs().mean()):.6f}')
