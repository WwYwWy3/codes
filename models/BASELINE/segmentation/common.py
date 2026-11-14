"""跨模型复用的输入预处理。"""
import torch


def to_three_channels(x: torch.Tensor) -> torch.Tensor:
    """输入只有 1 个通道时沿通道维复制 3 次，兼容按 3 通道构建的主干/预训练权重。"""
    if x.shape[1] == 1:
        return x.repeat(1, 3, *([1] * (x.dim() - 2)))
    return x
