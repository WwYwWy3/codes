import os
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch


class VISUALIZER:
    """把模型预测结果可视化并保存到 ``save_dir``。

    - 2D 数据集：每个 ``select_index`` 输出一张 (image | ground truth | prediction) 三联图
    - 3D 数据集（BTCV 这类按切片组织的）：每个 volume 输出
      1) 所有切片的分页三联图
      2) 用 pyvista 渲染的 3D 分割效果（``*_3d.png``）
      3) 可直接加载的体数据 ``*_volume.npz``
    pyvista 不可用（没装 / 没有 OpenGL 上下文）时自动退回 matplotlib 的 voxel 渲染。
    """

    slices_per_page = 8
    voxel_target = 40
    surface_target = 160
    pyvista_window_size = (1400, 700)

    def __init__(self, model, save_dir, dataset, select_index: Optional[List] = None,
                 device: str = 'cpu', class_nums: Optional[int] = None,
                 use_pyvista: bool = True, interactive: bool = False,
                 save_volume: bool = True) -> None:
        self.model = model
        self.save_dir = save_dir
        self.dataset = dataset
        self.device = device
        self.dataset_name = getattr(dataset, 'dataset_name', 'dataset')
        self.use_pyvista = use_pyvista
        self.interactive = interactive
        self.save_volume = save_volume

        config = getattr(dataset, 'config', None) or {}
        self.class_nums = class_nums or config.get('NUM_CLASSES', 2)  # 模型输出通道数
        self.num_labels = max(self.class_nums, 2)                     # 标签/配色数量（class_nums=1 时为背景+前景）
        self.palette = self._build_palette(config.get('COLOR', None))

        self.select_index = list(select_index) if select_index else self._default_select()

    # ------------------------------------------------------------------ 入口
    def __call__(self) -> str:
        self.model.eval()
        os.makedirs(self.save_dir, exist_ok=True)
        if self._is_3d():
            self._visualize_3d()
        else:
            self._visualize_2d()
        return self.save_dir

    def _is_3d(self) -> bool:
        data_list = getattr(self.dataset, 'data_list', []) or []
        return bool(data_list) and all('idx' in sample for sample in data_list)

    def _default_select(self) -> List[int]:
        if self._is_3d():
            volumes = sorted({sample['idx'] for sample in self.dataset.data_list})
            step = max(1, len(volumes) // 5)
            return volumes[::step][:5]
        length = len(self.dataset)
        step = max(1, length // 5)
        return list(range(0, length, step))[:5]

    # ------------------------------------------------------------------ 2D
    def _visualize_2d(self) -> None:
        for index in self.select_index:
            if index >= len(self.dataset):
                continue
            sample = self.dataset[index]
            pred = self._predict(sample['image'])
            panels = [
                (self._to_display(sample['image']), 'Image'),
                (self._colorize(sample['mask']), 'Ground truth'),
                (self._colorize(pred), 'Prediction'),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax, (data, title) in zip(axes, panels):
                ax.imshow(data)
                ax.set_title(title)
                ax.axis('off')
            fig.savefig(os.path.join(self.save_dir, f'{self.dataset_name}_{index}.png'),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)

    # ------------------------------------------------------------------ 3D
    def _visualize_3d(self) -> None:
        volume_slices = {}
        for index, sample in enumerate(self.dataset.data_list):
            volume_slices.setdefault(sample['idx'], []).append(index)

        for volume_index in self.select_index:
            if volume_index not in volume_slices:
                continue
            images, masks, preds = [], [], []
            for index in volume_slices[volume_index]:
                sample = self.dataset[index]
                images.append(sample['image'])
                masks.append(np.asarray(sample['mask']).astype(np.int64))
                preds.append(self._predict(sample['image']))

            tag = f'{self.dataset_name}_vol{volume_index}'
            image_volume = self._to_volume(images)
            gt_volume = np.stack(masks)
            pred_volume = np.stack(preds)

            self._save_slice_pages(volume_index, images, masks, preds)
            if self.save_volume:
                np.savez_compressed(os.path.join(self.save_dir, f'{tag}_volume.npz'),
                                    image=image_volume, gt=gt_volume, pred=pred_volume)

            if not (self.use_pyvista and self._render_pyvista(tag, gt_volume, pred_volume)):
                self._render_voxels(volume_index, gt_volume, pred_volume)

    def _to_volume(self, slices: List) -> np.ndarray:
        """[N, C, H, W] -> [N, H, W]（单通道）或 [N, H, W, C]，供体数据保存使用。"""
        volume = np.stack([np.asarray(item) for item in slices]).astype(np.float32)
        if volume.ndim == 4 and volume.shape[1] == 1:
            volume = volume[:, 0]
        elif volume.ndim == 4:
            volume = np.moveaxis(volume, 1, -1)
        return volume

    # ------------------------------------------------------------------ pyvista
    def _render_pyvista(self, tag: str, gt: np.ndarray, pred: np.ndarray) -> bool:
        """用 pyvista 离屏渲染 3D 分割效果并截图；失败返回 False。"""
        try:
            import pyvista as pv
        except ImportError:
            print('[VISUALIZER] 未安装 pyvista，改用 matplotlib 渲染')
            return False

        plotter = None
        try:
            if not self.interactive and not os.environ.get('DISPLAY'):
                pv.start_xvfb()
            plotter = pv.Plotter(off_screen=not self.interactive,
                                 window_size=self.pyvista_window_size, shape=(1, 2))
            self._add_pyvista_panel(pv, plotter, 0, 'Ground truth 3D', gt)
            self._add_pyvista_panel(pv, plotter, 1, 'Prediction 3D', pred)

            if self.interactive:
                plotter.show()
                return True

            screenshot = plotter.screenshot(return_img=True)
            # 无 OpenGL 上下文时截出来是全空图，这里判空后退回 matplotlib
            if screenshot is None or screenshot.size == 0 or np.ptp(screenshot) == 0:
                print('[VISUALIZER] pyvista 截图为空（缺少 OpenGL 上下文），改用 matplotlib 渲染')
                return False
            plt.imsave(os.path.join(self.save_dir, f'{tag}_3d.png'), screenshot)
            return True
        except Exception as exc:
            print(f'[VISUALIZER] pyvista 渲染失败({type(exc).__name__}: {exc})，改用 matplotlib 渲染')
            return False
        finally:
            if plotter is not None:
                try:
                    plotter.close()
                except Exception:
                    pass

    def _add_pyvista_panel(self, pv, plotter, column: int, title: str, volume: np.ndarray) -> None:
        """把每个类别的分割结果做等值面（marching cubes）后叠加到同一子图。"""
        plotter.subplot(0, column)
        plotter.set_background('white')
        plotter.add_text(title, font_size=10, color='black')

        volume = self._downsample(volume, self.surface_target)
        grid = pv.ImageData(dimensions=volume.shape)
        for class_index in range(1, self.num_labels):
            mask = volume == class_index
            if not mask.any():
                continue
            grid.point_data['mask'] = mask.ravel(order='F').astype(np.float32)
            surface = grid.contour([0.5], scalars='mask')
            plotter.add_mesh(surface, color=tuple(self.palette[class_index]),
                             opacity=0.75, smooth_shading=True, show_scalar_bar=False)

        plotter.view_isometric()

    # ------------------------------------------------------------------ 兜底渲染
    def _render_voxels(self, volume_index: int, gt_volume: np.ndarray, pred_volume: np.ndarray) -> None:
        targets = [('Ground truth 3D', self._downsample(gt_volume)),
                   ('Prediction 3D', self._downsample(pred_volume))]

        fig = plt.figure(figsize=(10, 5))
        for panel, (title, volume) in enumerate(targets, start=1):
            ax = fig.add_subplot(1, 2, panel, projection='3d')
            for class_index in range(1, self.num_labels):
                filled = volume == class_index
                if not filled.any():
                    continue
                ax.voxels(filled, facecolors=tuple(self.palette[class_index]), edgecolor=None)
            ax.set_title(title)
            ax.set_axis_off()

        fig.savefig(os.path.join(self.save_dir, f'{self.dataset_name}_vol{volume_index}_3d.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _downsample(self, volume: np.ndarray, target: Optional[int] = None) -> np.ndarray:
        target = target or self.voxel_target
        step = max(1, int(np.ceil(max(volume.shape) / target)))
        return volume[::step, ::step, ::step]

    def _save_slice_pages(self, volume_index: int, images: List, masks: List, preds: List) -> None:
        total = len(images)
        for page, start in enumerate(range(0, total, self.slices_per_page)):
            rows = list(range(start, min(start + self.slices_per_page, total)))
            fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
            for row, slice_index in enumerate(rows):
                panels = [
                    (self._to_display(images[slice_index]), f'slice {slice_index}'),
                    (self._colorize(masks[slice_index]), 'Ground truth'),
                    (self._colorize(preds[slice_index]), 'Prediction'),
                ]
                for col, (data, title) in enumerate(panels):
                    axes[row][col].imshow(data)
                    axes[row][col].set_title(title)
                    axes[row][col].axis('off')
            fig.savefig(os.path.join(self.save_dir, f'{self.dataset_name}_vol{volume_index}_slices_{page}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    # ------------------------------------------------------------------ 工具
    @torch.no_grad()
    def _predict(self, image) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] == 1:
            pred = (logits[:, 0].sigmoid() > 0.5).long()
        else:
            pred = logits.argmax(dim=1)
        return pred[0].cpu().numpy()

    def _build_palette(self, color) -> np.ndarray:
        if color:
            palette = np.zeros((self.num_labels, 3), dtype=np.float32)
            for index, value in color.items():
                if int(index) < self.num_labels:
                    palette[int(index)] = np.asarray(value, dtype=np.float32) / 255.0
            return palette
        colormap = plt.get_cmap('tab20')
        return np.asarray([colormap(i % 20)[:3] for i in range(self.num_labels)], dtype=np.float32)

    def _colorize(self, label) -> np.ndarray:
        label = np.clip(np.asarray(label).astype(np.int64), 0, self.num_labels - 1)
        return self.palette[label]

    @staticmethod
    def _to_display(image) -> np.ndarray:
        array = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        array = array.numpy()
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        array = array - array.min()
        return array / (array.max() + 1e-8)