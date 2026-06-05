import os
import numpy as np
import torch
from torch.utils import data
from data_augmentation import DataTransform3D


class SeisFaultOriginalDataset(data.dataloader.Dataset):

    def __init__(self,
                 dir=None,
                 set='train',
                 dim=(128, 128, 128),
                 dtype=np.single,
                 normalize=True,
                 transpose=True,
                 transform=None,
                 augment_prob=0.3):
        self.dir = dir
        self.set = set
        self.dim = dim
        self.dtype = dtype
        self.normalize = normalize
        self.transpose = transpose
        self.transform = transform
        if self.transform is None and set == 'train':
            self.transform = DataTransform3D(
                x_flip=True,
                y_flip=True,
                z_flip=True,
                rotate=True,
                rotation_planes=("xy",),
                erase=False,
                scale=False,
                noise=False,
                probability=augment_prob,
                erase_params=(0.02, 0.1, 0.3),
                scale_range=(0.9, 1.1),
                noise_std=0.30
            )

        self.seis_dir = os.path.join(dir, set, 'seis')
        self.fault_dir = os.path.join(dir, set, 'fault')

        self.files = []
        self._collect_files()

    def _collect_files(self):
        if not (os.path.isdir(self.seis_dir) and os.path.isdir(self.fault_dir)):
            return
        seis_ids = {os.path.splitext(f)[0] for f in os.listdir(self.seis_dir) if f.lower().endswith('.dat')}
        fault_ids = {os.path.splitext(f)[0] for f in os.listdir(self.fault_dir) if f.lower().endswith('.dat')}
        names = sorted(seis_ids & fault_ids, key=lambda x: (len(x), x))
        for name in names:
            seis_path = os.path.join(self.seis_dir, f'{name}.dat')
            fault_path = os.path.join(self.fault_dir, f'{name}.dat')
            if os.path.isfile(seis_path) and os.path.isfile(fault_path):
                self.files.append({'seis': seis_path, 'fault': fault_path, 'name': name})

    def __len__(self):
        return len(self.files)

    def _read_dat(self, path):
        arr = np.fromfile(path, dtype=self.dtype)
        arr = np.reshape(arr, self.dim)
        return arr

    def _zscore(self, x):
        xm = np.mean(x, dtype=np.float64)
        xs = np.std(x, dtype=np.float64)
        xs = xs if xs > 0 else 1.0
        return (x - xm) / xs

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']

        gx = self._read_dat(item['seis'])
        fx = self._read_dat(item['fault'])

        if self.normalize:
            gx = self._zscore(gx)

        # match Keras generator behavior: transpose (reverse axes)
        if self.transpose:
            gx = np.transpose(gx)
            fx = np.transpose(fx)

        # ensure contiguous before torch.from_numpy
        gx = np.ascontiguousarray(gx, dtype=np.float32)
        fx = np.ascontiguousarray(fx, dtype=np.float32)

        img = torch.from_numpy(gx).unsqueeze(0)   # (1, D, H, W)
        label = torch.from_numpy(fx).unsqueeze(0) # (1, D, H, W)

        # apply 3D data augmentation (probabilistic), if configured
        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label, name


