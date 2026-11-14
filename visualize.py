"""加载已训练的权重，对验证/测试集做可视化。

示例:
    python visualize.py --model_name unet --weights runs/btcv_unet/weights/unet_best.pth --dataset BTCV --save_dir runs/btcv_unet/visuals
"""
import argparse
import os

import torch

from collect_component import CLASS_NUM_KEYS, ROOT, _eval_augmentation, _override_param, load_yaml
from dataloader import create_dataloader
from models import create_baseline
from utils.visualizer import VISUALIZER


def parse_indices(text):
    if not text:
        return None
    return [int(item) for item in text.split(',')]


def run(args):
    model_configs = load_yaml(args.model_config).get('model_info', {})
    if args.model_name not in model_configs:
        raise KeyError(f'{args.model_name} 不在 {args.model_config} 中，可选: {sorted(model_configs)}')

    dataset_path = args.dataset if os.path.isfile(args.dataset) else os.path.join(ROOT, 'config', 'datasets', f'{args.dataset}.yaml')
    dataset_config = load_yaml(dataset_path)
    augmentation = load_yaml('/mnt/f/PatternRecognition/codes/config/datasets/aug_test_val.yaml')
    class_nums = dataset_config['DATASET']['NUM_CLASSES']

    model = create_baseline(
        args.model_name,
        _override_param(model_configs[args.model_name], class_nums, CLASS_NUM_KEYS, '类别数'),
    )

    state_dict = torch.load(args.weights, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        # TRAINER 保存的是包含优化器等信息的 checkpoint
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # 验证/测试集：同样的预处理，但随机增强全部关掉
    loader = create_dataloader('segmentation', args.split, dataset_config, _eval_augmentation(augmentation))
    save_dir = args.save_dir or os.path.join(ROOT, 'runs', 'visualize', args.model_name)

    visualizer = VISUALIZER(
        model, save_dir, loader.dataset,
        select_index=parse_indices(args.select_index), device=device,
        use_pyvista=not args.no_pyvista, interactive=args.interactive,
    )
    return visualizer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load trained weights and visualize predictions')
    parser.add_argument('--model_name', type=str, required=True, help='model name，与 config/models.yaml 一致')
    parser.add_argument('--weights', type=str, required=True, help='权重路径(.pth)')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name 或数据集配置路径')
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'], help='用哪个划分做可视化')
    parser.add_argument('--save_dir', type=str, default=None, help='输出目录，默认 runs/visualize/<model_name>')
    parser.add_argument('--select_index', type=str, default=None, help='逗号分隔的样本序号，3D 数据集为 volume 序号')
    parser.add_argument('--device', type=str, default=None, help='cpu / cuda, 默认自动选择')
    parser.add_argument('--model_config', type=str, default=os.path.join(ROOT, 'config', 'models.yaml'), help='模型配置')
    parser.add_argument('--interactive', action='store_true', help='打开 pyvista 交互窗口查看 3D 结果（需要显示器）')
    parser.add_argument('--no_pyvista', action='store_true', help='3D 用 matplotlib voxel 渲染，不调用 pyvista')

    print('saved to', run(parser.parse_args()))
