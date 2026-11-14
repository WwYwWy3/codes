import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

_PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'plots')

class TRAINER:
    def __init__(self, params: Dict, compare_fun: callable = None) -> None:
        self.model: nn.Module = params.get("model", None)
        self.model_name: str = params.get("model_name", None)
        self.optimizer: Optimizer = params.get("optimizer", None)
        self.lr_scheduler = params.get("lr_scheduler", None)
        self.criterion: nn.Module = params.get("criterion", None)
        self.train_loader: DataLoader = params.get("train_loader", None)
        self.validate_loader: DataLoader = params.get("validate_loader", None)
        self.validate_dataset = self.validate_loader.dataset if self.validate_loader is not None else None
        self.save_dir: str = params.get("weight_save_dir", None)
        self.evaluator = params.get("evaluator", None)
        self.device: str | torch.device | None = params.get("device", None)
        self.dataset_name: str = params.get('dataset_name', None)
        self.compare_fun = compare_fun

        self.weight_save_dir = os.path.join(self.save_dir, 'weights')
        self.plot_save_dir = os.path.join(self.save_dir, 'plots')
        self.visual_save_dir = os.path.join(self.save_dir, 'visuals')

        os.makedirs(self.weight_save_dir, exist_ok=True) 
        os.makedirs(self.plot_save_dir, exist_ok=True)

        self.train_len = len(self.train_loader)
        self.validate_len = len(self.validate_loader) if self.validate_loader is not None else 0

        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'matrics': [],
            'best_matrics': {}
        }

    def train_one_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0
        for data in self.train_loader:
            image, mask = data['image'].to(self.device), data['mask'].to(self.device)
            pred = self.model(image)
            loss = self.criterion(pred, mask)
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss / self.train_len

    def train(self, epochs: int = 300, eval_epochs: int = 1):
        self.model.to(self.device)
        self.training_history['config'] = {
            'dataset': self.dataset_name,
            'epochs': epochs,
            'eval_epochs': eval_epochs,
            'optimizer': type(self.optimizer).__name__,
            'lr': self.optimizer.param_groups[0]['lr'],
            'device': str(self.device),
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            self.training_history['train_loss'].append(train_loss)
            print(f'[epoch {epoch}/{epochs}] train_loss={train_loss:.4f}')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.validate_loader is not None and epoch % eval_epochs == 0:
                eval_matrics, eval_loss = self.evaluate()
                self.training_history['matrics'].append(eval_matrics)
                self.training_history['eval_loss'].append(eval_loss)
                self.print_metrics(epoch, epochs, eval_loss, eval_matrics)
                if self.compare_fun is None or self.compare_fun(eval_matrics, self.training_history['best_matrics']):
                    self.training_history['best_matrics'] = eval_matrics
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'config': self.training_history['config'],
                        'best_matrics': self.training_history['best_matrics'],
                    }, os.path.join(self.weight_save_dir, f'{self.model_name}_best.pth'))

        self.save_history()

        return self.training_history

    def print_metrics(self, epoch: int, epochs: int, eval_loss: float, eval_matrics: Dict) -> None:
        
        average = {name: value for name, value in eval_matrics.items()
                   if name not in ('per_class', 'in_gt')}
        print(f'[epoch {epoch}/{epochs}] eval_loss={eval_loss:.4f}')

        per_class = eval_matrics.get('per_class') or {}
        if not per_class:
            return

        metric_names = list(next(iter(per_class.values())))
        in_gt = eval_matrics.get('in_gt') or {}
        name_width = max([len(name) for name in per_class] + [5])
        print(f'{"class":<{name_width}} {"in_gt":>5}' + ''.join(f' {name:>8}' for name in metric_names))
        print('-' * (name_width + 6 + 9 * len(metric_names)))
        for class_name, scores in per_class.items():
            present = 'yes' if in_gt.get(class_name) else 'no'
            cells = ''.join(f' {self._format_score(scores[name]):>8}' for name in metric_names)
            print(f'{class_name:<{name_width}} {present:>5}{cells}')
        cells = ''.join(f' {self._format_score(average[name]):>8}' for name in metric_names)
        print(f'{"all":<{name_width}} {"":>5}{cells}')

    @staticmethod
    def _format_score(value: Optional[float]) -> str:
        return '-' if value is None else f'{value:.4f}'

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.evaluator.reset()
        eval_loss = 0.0
        if self.dataset_name != 'BTCV':
            for data in self.validate_loader:
                image, mask = data['image'].to(self.device), data['mask'].to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, mask)
                eval_loss += loss.item()
                self.evaluator.update(pred, mask)
        else:
            preds, masks = [], []
            cur_idx = None
            for data in self.validate_loader:
                if data['idx'].numel() != 1:
                    raise ValueError('BTCV 的 3D 评估要求 validate_loader 的 batch_size=1')
                image, mask = data['image'].to(self.device), data['mask'].to(self.device)
                idx = int(data['idx'].item())
                pred = self.model(image)
                eval_loss += self.criterion(pred, mask).item()

                # 切片按 volume 累积，遇到下一个 volume 时整体评估
                if cur_idx is not None and idx != cur_idx:
                    self._update_volume(preds, masks)
                    preds, masks = [], []
                preds.append(pred)
                masks.append(mask)
                cur_idx = idx
            # 最后一个 volume 不会再有分界点，需手动 flush
            if preds:
                self._update_volume(preds, masks)
        return self.evaluator.compute(), eval_loss / self.validate_len

    def _update_volume(self, preds: List[torch.Tensor], masks: List[torch.Tensor]):
        # 逐切片 [1, C, W, H] -> [1, C, D, W, H]; mask [1, W, H] -> [1, D, W, H]
        pred_3d = torch.cat(preds, dim=0)
        pred_3d = pred_3d.permute(1, 0, *range(2, pred_3d.dim())).unsqueeze(0)
        mask_3d = torch.cat(masks, dim=0).unsqueeze(0)
        self.evaluator.update(pred_3d, mask_3d)

    def save_history(self) -> str:
        """把训练历史写成 json，便于后续复现/画图。"""
        path = os.path.join(self.save_dir, f'{self.model_name}_{self.dataset_name}_training_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        return path