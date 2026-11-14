import copy
import os
from typing import Dict, Tuple

from dataloader import create_dataloader
from models import create_baseline
from models.BASELINE.segmentation.loss import LossFunction
from utils import Evaluator, TRAINER, create_optimizer_lr_scheduler

ROOT = os.path.dirname(os.path.abspath(__file__))
# 不同模型的类别数参数名不统一，按顺序取第一个命中的
CLASS_NUM_KEYS = ('class_nums', 'num_classes', 'out_channels')


def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _override_param(model_params: Dict, value: int, keys: tuple, name: str) -> Dict:
    params = dict(model_params)
    for key in keys:
        if key in params:
            params[key] = value
            return params
    raise KeyError(f'model config 里没有{name}参数，期望 {keys} 之一')


def _eval_augmentation(aug_config: Dict) -> Dict:
    """验证/可视化不需要随机增强，把所有随机操作的 p 置 0。"""
    config = copy.deepcopy(aug_config)
    for operation in config.get('OPERATIONS', {}).values():
        if 'p' in operation:
            operation['p'] = 0.0
    return config


class SEGCOLLECTOR:
    """按 (模型, 数据集, 优化器, 调度器) 组装出可直接训练的 TRAINER。"""
    def __call__(
        self,
        CONFIG: Dict
    ) -> Tuple[TRAINER, int, int]:
        paths = CONFIG['CONFIG_FILE_PATH']
        model_name, dataset_name = CONFIG['MODEL'], CONFIG['DATASET']
        epochs, eval_epoch = CONFIG['EPOCHS'], CONFIG['EVAL_EPOCH']

        dataset_config = load_yaml(paths['dataset'][dataset_name])
        class_nums = dataset_config['DATASET']['NUM_CLASSES']

        model_configs = load_yaml(paths['model']).get('model_info', {})
        if model_name not in model_configs:
            raise KeyError(f'{model_name} 不在 {paths["model"]} 中，可选: {sorted(model_configs)}')
        model = create_baseline(
            model_name,
            _override_param(model_configs[model_name], class_nums, CLASS_NUM_KEYS, '类别数'),
        )

        train_loader = create_dataloader(
            'segmentation', 'train', dataset_config, load_yaml(paths['augmentation']['train']),
        )
        validate_loader = create_dataloader(
            'segmentation', 'validation', dataset_config,
            _eval_augmentation(load_yaml(paths['augmentation']['test_val'])),
        )

        optimizer_config = load_yaml(paths['optimizer'])
        optimizer_name, scheduler_name = CONFIG['OPTIMIZER'], CONFIG['LR_SCHEDULER']
        # optimizer.yaml 里调度器的键名是 COSINEANNEALINGLR / STEP 这类写法
        scheduler_keys = (scheduler_name.upper(), f'{scheduler_name.upper()}LR')
        scheduler_config = next(
            (optimizer_config['LR_SCHEDULER'][key] for key in scheduler_keys
             if key in optimizer_config['LR_SCHEDULER']), None
        )
        if scheduler_config is None:
            raise KeyError(f'{scheduler_name} 不在 {paths["optimizer"]} 的 LR_SCHEDULER 中，'
                           f'可选: {sorted(optimizer_config["LR_SCHEDULER"])}')
        scheduler_params = dict(scheduler_config)
        # 配置里留空的周期参数按总 epoch 推导，避免直接传给 torch 报 None
        if scheduler_name.lower() == 'step' and scheduler_params.get('step_size') is None:
            scheduler_params['step_size'] = max(1, epochs // 10)
        if scheduler_name.lower().startswith('cosine') and scheduler_params.get('T_max') is None:
            scheduler_params['T_max'] = epochs
        optimizer, lr_scheduler = create_optimizer_lr_scheduler(
            optimizer_name, scheduler_name, model.parameters(),
            optimizer_config['OPTIMIZER'][optimizer_name.upper()], scheduler_params,
        )

        criterion = LossFunction(w_dice=CONFIG['W_DICE'], w_ce=CONFIG['W_CE'], class_nums=class_nums)
        # 逐类指标使用当前数据集自身配置的类别：BTCV -> 13 个器官，BUSI -> ill
        evaluator = Evaluator(
            class_nums=class_nums,
            class_names=dataset_config['DATASET'].get('LABEL'),
        )

        trainer = TRAINER(
            {
                'model': model,
                'model_name': model_name,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'criterion': criterion,
                'train_loader': train_loader,
                'validate_loader': validate_loader,
                'weight_save_dir': os.path.join(CONFIG['SAVE_DIR'], f'{dataset_name}_{model_name}'),
                'evaluator': evaluator,
                'device': CONFIG.get('DEVICE', 'cuda'),
                'dataset_name': dataset_name,
                'select_index': CONFIG.get('SELECT_INDEX', None),
            },
            # 统一用所有类别的平均 dice 比较，dice 越高越好
            compare_fun=lambda cur, best: cur['dice'] > best.get('dice', -1),
        )
        return trainer, epochs, eval_epoch

