import os
import sys
from typing import Any, Dict, Tuple

from monai.metrics import DiceMetric, MeanIoU, SurfaceDistanceMetric
import numpy as np
import torch
import torch.nn.functional as F

def calc_asd(pred: torch.Tensor, gt: torch.Tensor, class_nums: int, include_background: bool = True, symmetric: bool = True, distance_metric = "euclidean", reduction: str = "mean"):
    asd_metric = SurfaceDistanceMetric(
        include_background=include_background,
        symmetric=symmetric,          # 是否计算对称 ASD，建议 True
        distance_metric=distance_metric,
        reduction=reduction
    )
    gt = transform_multi_to_onehot(gt, class_nums)
    asd_metric(y_pred = pred, y = gt)
    result = asd_metric.aggregate().item()
    return result

def calc_iou(pred: torch.Tensor, gt: torch.Tensor, class_nums: int, include_background: bool = True, reduction: str = "mean"):
    IoU_metric = MeanIoU(include_background=include_background, reduction=reduction)
    gt = transform_multi_to_onehot(gt, class_nums)
    IoU_metric(y_pred = pred, y = gt)
    result = IoU_metric.aggregate().item()
    return result

def calc_dice(pred: torch.Tensor, gt: torch.Tensor, class_nums: int, have_channel: bool = True, include_background: bool = False, reduction: str = "mean"):
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    gt = transform_multi_to_onehot(gt, class_nums, have_channel)
    dice_metric(y_pred = pred, y = gt)
    result = dice_metric.aggregate().item()
    return result

def transform_multi_to_onehot(gt: torch.Tensor, class_nums: int, have_channel: bool = True):
    # gt: [batch, 1, ...] 或 [batch, ...]，2D / 3D 通用
    if have_channel:
        gt_ = gt.squeeze(1).to(torch.long)
    else:
        gt_ = gt.to(torch.long)
    gt_onehot = F.one_hot(gt_, num_classes = class_nums).movedim(-1, 1).float() # [batch, class_nums, ...]
    return gt_onehot

class Evaluator:
    """整体 + 逐类指标（均不含 background）。

    usage:
        # class_names 用当前数据集配置里的 LABEL：BTCV -> 13 个器官，BUSI -> ill
        ev = Evaluator(class_nums=dataset_config['DATASET']['NUM_CLASSES'],
                       class_names=dataset_config['DATASET']['LABEL'])
        ev.update(pred, gt)     # pred: [B, C, ...] logits; gt: [B, ...] 或 [B, 1, ...] 标签图
        metrics = ev.compute()  # {'dice': ..., 'iou': ...,
                                #  'per_class': {'<数据集里的类别名>': {'dice': ..., 'iou': ...}, ...},
                                #  'in_gt': {'<数据集里的类别名>': True/False, ...}}
        ev.reset()

    dice/iou 是所有前景类的平均值；某个类在所有样本里都没出现时该类为 null，且不参与平均。
    in_gt 表示该类是否在标注中出现过。
    """

    def __init__(self, class_nums: int, class_names: Dict = None, gt_have_channel: bool = False, metrics: Tuple[str, ...] = ("dice", "iou")) -> None:
        self.class_nums = class_nums
        self.num_labels = max(class_nums, 2)  # class_nums == 1 时标签为 背景+前景
        self.gt_have_channel = gt_have_channel
        self.class_names = {index: self._class_name(index, class_names) for index in range(1, self.num_labels)}
        self.gt_counts = torch.zeros(self.num_labels, dtype=torch.long)  # 各类在标注中出现的像素/体素数
        self.metrics: Dict[str, Any] = {}
        if "dice" in metrics:
            # include_background=False 会去掉第 0 通道；reduction="none" 保留每类数值，便于逐类统计
            self.metrics["dice"] = DiceMetric(include_background=False, reduction="none")
        if "iou" in metrics:
            self.metrics["iou"] = MeanIoU(include_background=False, reduction="none")

    @staticmethod
    def _class_name(index: int, class_names: Dict) -> str:
        if not class_names:
            return str(index)
        return str(class_names.get(index, class_names.get(str(index), index)))

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        # pred logits [B, C, ...] -> 类别索引 [B, ...]；class_nums == 1 时按 sigmoid 0.5 二值化
        if self.class_nums == 1:
            label = (pred[:, 0].sigmoid() > 0.5).long()
        else:
            label = pred.argmax(dim=1)
        y_pred = transform_multi_to_onehot(label, self.num_labels)
        y = transform_multi_to_onehot(gt, self.num_labels, self.gt_have_channel)
        self.gt_counts += y.flatten(2).sum(dim=(0, 2)).long().cpu()
        for metric in self.metrics.values():
            metric(y_pred=y_pred, y=y)

    def compute(self) -> Dict[str, Any]:
        per_class = {name: {} for name in self.class_names.values()}
        average = {}
        for metric_name, metric in self.metrics.items():
            if metric.get_buffer() is None:  # 还没有 update 过
                scores = [float("nan")] * (self.num_labels - 1)
            else:
                scores = self._nan_mean(metric.aggregate()).tolist()  # [C-1]，未出现的类别为 nan

            valid = [score for score in scores if not np.isnan(score)]
            average[metric_name] = float(np.mean(valid)) if valid else float("nan")
            for name, score in zip(self.class_names.values(), scores):
                # 缺失类别用 null(json) 表示
                per_class[name][metric_name] = None if np.isnan(score) else float(score)
        in_gt = {name: bool(self.gt_counts[index]) for index, name in self.class_names.items()}
        return {**average, "per_class": per_class, "in_gt": in_gt}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.gt_counts.zero_()

    @staticmethod
    def _nan_mean(values: torch.Tensor) -> torch.Tensor:
        # 按类对所有样本求均值，整类都是 nan 时保持 nan
        valid = ~torch.isnan(values)
        counts = valid.sum(dim=0)
        sums = torch.where(valid, values, torch.zeros_like(values)).sum(dim=0)
        means = sums / counts.clamp(min=1)
        return torch.where(counts > 0, means, torch.full_like(means, float("nan")))

if __name__ == '__main__':
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    path = '/mnt/f/PatternRecognition/codes/datasets/BTCV/processed_data/mask/label0026/75.npz'
    x = torch.tensor(np.load(path)['mask']).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1, 1) # (512, 512)
    y = transform_multi_to_onehot(x, 14)
    res = calc_dice(y, x, 14)
    print(res)

    


