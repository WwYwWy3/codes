import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

class SwinTransformer(nn.Module):
    def __init__(self, model_name: str = 'swin_v1_base', class_nums: int = 1000, pretrained: bool = False):
        super().__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        self.class_nums = class_nums
        self.model, self.classifier = self.__build_model__(model_name, class_nums, pretrained)

    def __build_model__(self, model_name: str, class_nums: int, pretrained: bool = False):
        if model_name == 'swin_v1_tiny':
            model = nn.Sequential(*list(models.swin_t(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(768, class_nums)
        elif model_name == 'swin_v1_small':
            model = nn.Sequential(*list(models.swin_s(pretrained=pretrained).children())[:-1])  
            classifier = nn.Linear(768, class_nums)
        elif model_name == 'swin_v1_base':
            model = nn.Sequential(*list(models.swin_b(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(1024, class_nums)
        elif model_name == 'swin_v2_tiny':
            model = nn.Sequential(*list(models.swin_v2_t(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(768, class_nums)
        elif model_name == 'swin_v2_small':
            model = nn.Sequential(*list(models.swin_v2_s(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(768, class_nums)
        elif model_name == 'swin_v2_base':
            model = nn.Sequential(*list(models.swin_v2_b(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(1024, class_nums)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model, classifier

    def GradCAM_forward(self, x: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model_name = ['swin_v1_tiny', 'swin_v1_small', 'swin_v1_base', 'swin_v2_tiny', 'swin_v2_small', 'swin_v2_base']
    for name in model_name:
        model = SwinTransformer(model_name=name, class_nums=50, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        print(model(x).shape)
