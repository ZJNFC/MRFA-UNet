import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'data'))

from data_loader import SeisFaultOriginalDataset
from MRFA_UNet import MRFA_UNet
from segnet import SegNet3D
from transunet import TransUNet
from unet import UNet3D

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_type = 'mrfa_unet'
model_weight = r''
model = None
val_output = None
F3_output = None

F3_DATASET = {
    'name': 'F3',
    'seis_dir': '',
    'fault_dir': '',
    'volume_shape': (512, 384, 128),
    'transpose_axes': None,
    'axis_name_map': {
        'x': ('vertical', 'crossline'),
        'y': ('vertical', 'inline'),
        'z': ('crossline', 'inline'),
    },
}


def _get_f3_names():
    seis_dir = F3_DATASET['seis_dir']
    fault_dir = F3_DATASET['fault_dir']
    if not seis_dir or not fault_dir:
        raise ValueError('Please set F3_DATASET[\'seis_dir\'] and F3_DATASET[\'fault_dir\'] before running F3 visualization')
    if not (os.path.isdir(seis_dir) and os.path.isdir(fault_dir)):
        raise FileNotFoundError(f'F3 data directories not found: seis_dir={seis_dir}, fault_dir={fault_dir}')

    seis_ids = {os.path.splitext(f)[0] for f in os.listdir(seis_dir) if f.lower().endswith('.dat')}
    fault_ids = {os.path.splitext(f)[0] for f in os.listdir(fault_dir) if f.lower().endswith('.dat')}
    return sorted(seis_ids & fault_ids, key=lambda x: (len(x), x))


def _load_f3_volume(name):
    seis_path = os.path.join(F3_DATASET['seis_dir'], f'{name}.dat')
    if not os.path.isfile(seis_path):
        raise FileNotFoundError(f'F3 seismic file not found: {seis_path}')
    arr = np.fromfile(seis_path, dtype=np.float32)
    arr = arr.reshape(F3_DATASET['volume_shape'])

    mean = np.mean(arr)
    std = np.std(arr)
    arr = (arr - mean) / std if std > 0 else arr - mean

    transpose_axes = F3_DATASET.get('transpose_axes')
    return np.transpose(arr) if transpose_axes is None else np.transpose(arr, transpose_axes)


def make_dir():
    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.date().strftime('%Y_%m_%d')
    formatted_time = current_datetime.time().strftime('%H_%M_%S')

    base_dir = os.path.join('result', model_type, 'output', formatted_date, formatted_time)
    current_val_output = os.path.join(base_dir, 'val')
    current_f3_output = os.path.join(base_dir, 'F3')

    os.makedirs(current_val_output, exist_ok=True)
    os.makedirs(current_f3_output, exist_ok=True)
    return current_val_output, current_f3_output


def build_model(current_model_type):
    if current_model_type == 'unet':
        current_model = UNet3D(in_channels=1, num_classes=1, features=(16, 32, 64, 128))
    elif current_model_type == 'mrfa_unet':
        current_model = MRFA_UNet(in_channels=1, num_classes=1, features=(16, 32, 64, 128))
    elif current_model_type == 'transunet':
        current_model = TransUNet(in_channels=1, num_classes=1)
    elif current_model_type == 'segnet':
        current_model = SegNet3D(in_channels=1, num_classes=1, features=[16, 32, 64, 128])
    else:
        raise ValueError('no this model')
    return current_model.to(device)


def load_single_weight(weight_path):
    if not weight_path:
        raise ValueError('model_weight is empty')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()


def _model_forward_probs(img):
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        logits = model(img)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        return torch.sigmoid(logits)


def _take_slice_3d(vol, axis, idx):
    depth, height, width = vol.shape
    if axis == 'z':
        if idx is None:
            idx = depth // 2
        idx = max(0, min(depth - 1, idx))
        return vol[idx], idx, 'z'
    if axis == 'y':
        if idx is None:
            idx = height // 2
        idx = max(0, min(height - 1, idx))
        return vol[:, idx, :], idx, 'y'
    if idx is None:
        idx = width // 2
    idx = max(0, min(width - 1, idx))
    return vol[:, :, idx], idx, 'x'


def _get_axes_list(slice_axes):
    if isinstance(slice_axes, str):
        return [slice_axes]
    if slice_axes is None:
        return ['z']
    return list(slice_axes)


def _get_index_list(slice_indices, axis):
    if slice_indices is None:
        return [None]
    if isinstance(slice_indices, dict):
        indices = slice_indices.get(axis, [None])
        return list(indices) if isinstance(indices, (list, tuple)) else [indices]
    return list(slice_indices) if isinstance(slice_indices, (list, tuple)) else [slice_indices]


def _set_compact_axis_labels(ax, xlabel=None, ylabel=None):
    font_properties = {'family': 'Times New Roman', 'size': 14}

    if xlabel:
        ax.text(0.5, 1.008, f'{xlabel}', transform=ax.transAxes, ha='center', va='bottom', **font_properties)
    if ylabel:
        ax.text(1.002, 0.5, f'{ylabel}', transform=ax.transAxes, ha='left', va='center', rotation=-90, **font_properties)


def _format_axis_ticks(ax):
    ax.tick_params(axis='both', labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Times New Roman')


def _normalize_base_image(base_img_2d):
    mean = float(np.mean(base_img_2d))
    std = float(np.std(base_img_2d))
    if std > 0:
        return (base_img_2d - mean) / std + 0.7
    return np.zeros_like(base_img_2d, dtype=np.float32)


def _save_raw_image(base_img_2d, out_path, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_normalize_base_image(base_img_2d), cmap='bone', vmin=-3.0, vmax=3.0)
    _format_axis_ticks(ax)
    _set_compact_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    plt.subplots_adjust(left=0.02, right=0.97, bottom=0.02, top=0.96)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _save_overlay_binary(base_img_2d, mask_2d, out_path, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_normalize_base_image(base_img_2d), cmap='bone', vmin=-3.0, vmax=3.0)

    mask = (mask_2d > 0).astype(np.float32)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask * 0.95
    ax.imshow(overlay)

    _format_axis_ticks(ax)
    _set_compact_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    plt.subplots_adjust(left=0.02, right=0.97, bottom=0.02, top=0.96)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _save_overlay_heatmap(base_img_2d, prob_2d, out_path, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_normalize_base_image(base_img_2d), cmap='bone', vmin=-3.0, vmax=3.0)

    prob = np.clip(prob_2d.astype(np.float32), 0.0, 1.0)
    prob_color = prob ** 0.6
    alpha_map = np.zeros_like(prob_color, dtype=np.float32)
    alpha_map[prob > 0.1] = 0.95
    rgba = plt.get_cmap('turbo')(prob_color)
    rgba[..., :3] = np.clip(rgba[..., :3] * 2.0, 0, 1)
    rgba[..., 3] = alpha_map
    ax.imshow(rgba)

    _format_axis_ticks(ax)
    _set_compact_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    plt.subplots_adjust(left=0.02, right=0.97, bottom=0.02, top=0.96)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def val_data_overlay_label(slice_axis='z', slice_index=None):
    dataset = SeisFaultOriginalDataset(dir='data', set='validation', dim=(128, 128, 128), normalize=False, transpose=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for img, label, name in loader:
        img_np = img.cpu().squeeze(0).squeeze(0).numpy()
        label_np = label.cpu().squeeze(0).squeeze(0).numpy()

        slc_img, index, axis = _take_slice_3d(img_np, slice_axis, slice_index)
        slc_label, _, _ = _take_slice_3d(label_np, axis, index)
        out_path = os.path.join(val_output, f'{model_type}_{name[0]}_{axis}{index}_overlay_label.png')
        _save_overlay_binary(slc_img, slc_label, out_path)
        print(f'{name[0]} -> saved {os.path.basename(out_path)} | shape={img_np.shape} axis={axis} index={index}')


def val_data_overlay_heatmap(slice_axis='z', slice_index=None):
    dataset = SeisFaultOriginalDataset(dir='data', set='validation', dim=(128, 128, 128), normalize=False, transpose=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.inference_mode():
        for img, label, name in loader:
            img_tensor = img.to(device, non_blocking=True)
            probs = _model_forward_probs(img_tensor)
            prob_np = probs.float().cpu().squeeze(0).squeeze(0).numpy()
            img_np = img.cpu().squeeze(0).squeeze(0).numpy()

            slc_img, index, axis = _take_slice_3d(img_np, slice_axis, slice_index)
            slc_prob, _, _ = _take_slice_3d(prob_np, axis, index)
            out_path = os.path.join(val_output, f'{model_type}_{name[0]}_{axis}{index}_overlay_heatmap.png')
            _save_overlay_heatmap(slc_img, slc_prob, out_path)
            print(f'{name[0]} -> saved {os.path.basename(out_path)} | shape={img_np.shape} axis={axis} index={index}')

            del probs, img_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def F3_data_overlay_pred(threshold=0.9, slice_axes=None, slice_indices=None):
    names = _get_f3_names()
    axis_name_map = F3_DATASET['axis_name_map']

    model.eval()
    with torch.inference_mode():
        for name in names:
            arr = _load_f3_volume(name)
            img = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
            probs = _model_forward_probs(img)
            pred_np = (probs > threshold).float().cpu().squeeze(0).squeeze(0).numpy()

            del probs, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for axis in _get_axes_list(slice_axes):
                for index in _get_index_list(slice_indices, axis):
                    slc_img, index_eff, axis_eff = _take_slice_3d(arr, axis, index)
                    slc_pred, _, _ = _take_slice_3d(pred_np, axis_eff, index_eff)
                    xlabel, ylabel = axis_name_map.get(axis_eff, (None, None))
                    out_path = os.path.join(F3_output, f'{model_type}_{name}_{axis_eff}{index_eff}_overlay_pred_thr{threshold}.png')
                    _save_overlay_binary(slc_img, slc_pred, out_path, xlabel=xlabel, ylabel=ylabel)
                    print(f"[F3] {name} -> saved {os.path.basename(out_path)} | shape={arr.shape} axis={axis_eff} index={index_eff}")


def F3_data_overlay_heatmap(slice_axes=None, slice_indices=None):
    names = _get_f3_names()
    axis_name_map = F3_DATASET['axis_name_map']

    model.eval()
    with torch.inference_mode():
        for name in names:
            arr = _load_f3_volume(name)
            img = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
            probs = _model_forward_probs(img)
            prob_np = probs.float().cpu().squeeze(0).squeeze(0).numpy()

            del probs, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for axis in _get_axes_list(slice_axes):
                for index in _get_index_list(slice_indices, axis):
                    slc_img, index_eff, axis_eff = _take_slice_3d(arr, axis, index)
                    slc_prob, _, _ = _take_slice_3d(prob_np, axis_eff, index_eff)
                    xlabel, ylabel = axis_name_map.get(axis_eff, (None, None))
                    out_path = os.path.join(F3_output, f'{model_type}_{name}_{axis_eff}{index_eff}_overlay_heatmap.png')
                    _save_overlay_heatmap(slc_img, slc_prob, out_path, xlabel=xlabel, ylabel=ylabel)
                    print(f"[F3] {name} -> saved {os.path.basename(out_path)} | shape={arr.shape} axis={axis_eff} index={index_eff}")


def F3_data_raw(slice_axes=None, slice_indices=None):
    names = _get_f3_names()
    axis_name_map = F3_DATASET['axis_name_map']

    for name in names:
        arr = _load_f3_volume(name)
        for axis in _get_axes_list(slice_axes):
            for index in _get_index_list(slice_indices, axis):
                slc_img, index_eff, axis_eff = _take_slice_3d(arr, axis, index)
                xlabel, ylabel = axis_name_map.get(axis_eff, (None, None))
                out_path = os.path.join(F3_output, f'{model_type}_{name}_{axis_eff}{index_eff}_raw.png')
                _save_raw_image(slc_img, out_path, xlabel=xlabel, ylabel=ylabel)
                print(f"[F3] {name} -> saved {os.path.basename(out_path)} | shape={arr.shape} axis={axis_eff} index={index_eff}")


if __name__ == '__main__':
    model_type = 'mrfa_unet'
    model_weight = r''
    AXES = ['z']
    INDICES = {'z': [110]}

    model = build_model(model_type)
    val_output, F3_output = make_dir()
    load_single_weight(model_weight)

    F3_data_overlay_heatmap(slice_axes=AXES, slice_indices=INDICES)
