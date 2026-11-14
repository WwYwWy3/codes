import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

class ResNet(nn.Module):
    def __init__(self, model_name: str = 'resnet50', class_nums: int = 1000, pretrained: bool = False):
        super().__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        self.class_nums = class_nums
        self.model, self.classifier = self.__build_model__(model_name, class_nums, pretrained)

    def __build_model__(self, model_name: str, class_nums: int, pretrained: bool = False):
        if model_name == 'resnet18':
            model = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(512, class_nums)
        elif model_name == 'resnet34':
            model = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(512, class_nums)
        elif model_name == 'resnet50':
            model = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(2048, class_nums)
        elif model_name == 'resnet101':
            model = nn.Sequential(*list(models.resnet101(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(2048, class_nums)
        elif model_name == 'resnet152':
            model = nn.Sequential(*list(models.resnet152(pretrained=pretrained).children())[:-1])
            classifier = nn.Linear(2048, class_nums)
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
    model_name = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for name in model_name:
        model = ResNet(model_name=name, class_nums=50)
        x = torch.randn(4, 3, 224, 224)
        print(model(x).shape)

