import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

class ConvNeXt(nn.Module):
    def __init__(self, model_name: str, class_nums: int = 1000, pretrained: bool = False):
        super().__init__()
        self.model_name = model_name
        self.class_nums = class_nums
        self.pretrained = pretrained
        self.model = self.__build_model__(model_name, pretrained)
        self.__replace_head__(class_nums)

    def __build_model__(self, model_name: str, pretrained: bool = False):
        if model_name == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=pretrained)
        elif model_name == 'convnext_small':
            model = models.convnext_small(pretrained=pretrained)
        elif model_name == 'convnext_base':
            model = models.convnext_base(pretrained=pretrained)
        elif model_name == 'convnext_large':
            model = models.convnext_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model

    def __replace_head__(self, class_nums: int):
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, class_nums)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == '__main__':
    model_names = ['convnext_tiny']
    from torchinfo import summary
    for name in model_names:
        convnext = ConvNeXt(model_name=name, class_nums=50)
        x = torch.randn(4, 3, 224, 224)
        summary(convnext, input_data = x)
