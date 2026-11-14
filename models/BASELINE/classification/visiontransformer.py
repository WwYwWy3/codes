import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

try:
    import timm
except Exception as e:
    print(e)

import warnings
warnings.filterwarnings("ignore")

class VisionTransformer(nn.Module):
    def __init__(self, model_name: str, class_nums: int = 1000, pretrained: bool = False):
        super().__init__()
        self.model_name = model_name
        self.class_nums = class_nums
        self.pretrained = pretrained
        self.model = self.__build_model__(model_name, pretrained)
        self.__replace_head__(class_nums)

    def __build_model__(self, model_name: str, pretrained: bool = False):
        if model_name == 'vit_base_p16':
            model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_base_p32':
            model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_large_p16':
            model = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_large_p32':
            model = models.vit_l_32(pretrained=pretrained)
        elif model_name == 'vit_huge_p14':
            model = models.vit_h_14(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model

    def __replace_head__(self, class_nums: int):
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, class_nums)

    def GradCAM_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == '__main__':
    model_names = ['vit_base_p16', 'vit_base_p32', 'vit_large_p16', 'vit_large_p32', 'vit_huge_p14']
    for name in model_names:
        vit = VisionTransformer(model_name=name, class_nums=50)
        x = torch.randn(2, 3, 224, 224)
        out = vit(x)
        print(out.shape)
