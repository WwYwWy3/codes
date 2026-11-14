import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

class VGG(nn.Module):
    def __init__(self, model_name: str = 'vgg16', class_nums: int = 1000, pretrained: bool = False):
        super().__init__()
        self.model_name = model_name
        self.class_nums = class_nums
        self.pretrained = pretrained

        self.model, self.classifier = self.__build_model__(model_name, class_nums, pretrained)

    def __build_model__(self, model_name: str, class_nums: int, pretrained: bool = False):
        if model_name == 'vgg11':
            model = nn.Sequential(*list(models.vgg11_bn(pretrained=pretrained).children())[:-1])
        elif model_name == 'vgg13':
            model = nn.Sequential(*list(models.vgg13_bn(pretrained=pretrained).children())[:-1])
        elif model_name == 'vgg16':
            model = nn.Sequential(*list(models.vgg16_bn(pretrained=pretrained).children())[:-1])
        elif model_name == 'vgg19':
            model = nn.Sequential(*list(models.vgg19_bn(pretrained=pretrained).children())[:-1])
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5, inplace = False),
            nn.Linear(4096, 4096, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5, inplace = False),
            nn.Linear(4096, class_nums, bias = True)
        )

        return model, classifier

    def GradCAM_forward(self, x: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model_names = ['vgg11', 'vgg16', 'vgg19']
    for name in model_names:
        vgg = VGG(model_name=name, class_nums=50)
        x = torch.randn(4, 3, 224, 224)
        output = vgg(x)
        print(output.shape)
        print("\n")
        print("="*50)