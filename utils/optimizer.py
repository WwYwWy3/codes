import os
import sys
from typing import Dict

import torch.optim as optim

def create_optimizer(name: str = 'adamw', model_params=None, params: Dict = None):
    name_op = name.lower()
    config = dict(params or {})
    config['params'] = model_params

    if name_op == 'adam':
        return optim.Adam(**config)
    elif name_op == 'adamw':
        return optim.AdamW(**config)
    elif name_op == 'sgd':
        return optim.SGD(**config)
    else:
        raise ValueError(f'Unknown optimizer name: {name_op}')

def create_lr_scheduler(name: str = 'CosineAnnealing', optimizer: optim.Optimizer = None, params: Dict = None):
    name_op = name.lower()

    if name_op == 'cosineannealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name_op == 'cosineannealingwarmrestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    elif name_op == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif name_op == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        raise ValueError(f'Unknown lr scheduler name: {name_op}')

def create_optimizer_lr_scheduler(name_op, name_lr, model_params, params_op, params_lr):
    optimizer = create_optimizer(name_op, model_params, params_op)
    lr_scheduler = create_lr_scheduler(name_lr, optimizer, params_lr)

    return optimizer, lr_scheduler


