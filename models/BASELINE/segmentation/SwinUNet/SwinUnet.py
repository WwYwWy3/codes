from .vision_transformer import SwinUnet
from .config import get_config
import argparse

def swinunet(img_size=224, num_classes = 1, input_channel=3):
    model = SwinUnet(img_size=img_size, input_channel=input_channel, num_classes=num_classes)
    return model