import random

import torch
import torch.nn.functional as F


class DataTransform3D:
    def __init__(self,
                 x_flip=True,
                 y_flip=True,
                 z_flip=True,
                 rotate=True,
                 rotation_planes=("xy"),
                 erase=True,
                 scale=True,
                 noise=False,
                 probability=0.3,
                 erase_params=(0.05, 0.2, 0.3),
                 scale_range=(0.8, 1.2),
                 noise_std=0.05):

        self.x_flip = x_flip
        self.y_flip = y_flip
        self.z_flip = z_flip
        self.rotate = rotate
        self.rotation_planes = tuple(rotation_planes) if rotation_planes is not None else ("xy", "xz", "yz")
        self.erase = erase
        self.scale = scale
        self.noise = noise
        self.probability = probability

        if erase_params[0] > erase_params[1]:
            raise ValueError("Erase volume lower bound cannot be greater than upper bound")
        self.sl, self.sh, self.r1 = erase_params

        if scale_range[0] > scale_range[1]:
            raise ValueError("Minimum scale factor cannot be greater than maximum scale factor")
        self.scale_range = scale_range

        if noise_std < 0:
            raise ValueError("Noise standard deviation cannot be negative")
        self.noise_std = noise_std

    def _apply_to_all(self, args, func, indices=None):
        if indices is None:
            indices = range(len(args))
        return [func(arg) if i in indices else arg for i, arg in enumerate(args)]

    def __call__(self, *args):
        args = list(args)
        if not args:
            return tuple(args)

        c, d, h, w = args[0].shape

        if self.x_flip and random.random() < self.probability:
            flip_x = lambda x: torch.flip(x, dims=[3])
            args = self._apply_to_all(args, flip_x)

        if self.y_flip and random.random() < self.probability:
            flip_y = lambda x: torch.flip(x, dims=[2])
            args = self._apply_to_all(args, flip_y)

        if self.z_flip and random.random() < self.probability:
            flip_z = lambda x: torch.flip(x, dims=[1])
            args = self._apply_to_all(args, flip_z)

        if self.rotate and random.random() < self.probability:
            args = self._apply_rotation(args)

        if self.erase and len(args) > 1 and random.random() < self.probability:
            args = self._apply_erase_3d(args, d, h, w)

        if self.scale and random.random() < self.probability:
            args = self._apply_scale_crop_3d(args, d, h, w)

        if self.noise and len(args) > 1 and random.random() < self.probability:
            args = self._apply_noise_3d(args)

        return tuple(args)

    def _apply_rotation(self, args):
        plane = random.choice(self.rotation_planes)
        angle = random.choice([90, 180, 270])

        if plane == 'xy':
            if angle == 90:
                rotate_fn = lambda x: torch.rot90(x, k=1, dims=[2, 3])
            elif angle == 180:
                rotate_fn = lambda x: torch.rot90(x, k=2, dims=[2, 3])
            else:
                rotate_fn = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        elif plane == 'xz':
            if angle == 90:
                rotate_fn = lambda x: torch.rot90(x, k=1, dims=[1, 3])
            elif angle == 180:
                rotate_fn = lambda x: torch.rot90(x, k=2, dims=[1, 3])
            else:
                rotate_fn = lambda x: torch.rot90(x, k=3, dims=[1, 3])
        else:
            if angle == 90:
                rotate_fn = lambda x: torch.rot90(x, k=1, dims=[1, 2])
            elif angle == 180:
                rotate_fn = lambda x: torch.rot90(x, k=2, dims=[1, 2])
            else:
                rotate_fn = lambda x: torch.rot90(x, k=3, dims=[1, 2])

        return self._apply_to_all(args, rotate_fn)

    def _apply_erase_3d(self, args, d, h, w):
        image_args = args[:-1]
        label = args[-1]

        target_volume = random.uniform(self.sl, self.sh) * d * h * w
        aspect_ratio = random.uniform(1 / self.r1, self.r1)

        side_length = target_volume ** (1 / 3)
        de = int(round(side_length))
        he = int(round(side_length * aspect_ratio))
        we = int(round(side_length / aspect_ratio))

        de = max(1, min(de, d - 1))
        he = max(1, min(he, h - 1))
        we = max(1, min(we, w - 1))

        if de > 0 and he > 0 and we > 0:
            i = random.randint(0, d - de)
            j = random.randint(0, h - he)
            k = random.randint(0, w - we)

            erase_mask = torch.zeros(1, d, h, w, device=args[0].device)
            erase_mask[:, i:i + de, j:j + he, k:k + we] = 1

            return [img * (1 - erase_mask) for img in image_args] + [label]

        return args

    def _apply_scale_crop_3d(self, args, orig_d, orig_h, orig_w):
        scale = random.uniform(*self.scale_range)
        new_d = int(orig_d * scale)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        new_d = max(new_d, orig_d)
        new_h = max(new_h, orig_h)
        new_w = max(new_w, orig_w)

        scaled = []
        for idx, tensor in enumerate(args):
            mode = 'trilinear' if idx < len(args) - 1 else 'nearest'

            scaled_tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(new_d, new_h, new_w),
                mode=mode,
                align_corners=False if mode != 'nearest' else None
            ).squeeze(0)
            scaled.append(scaled_tensor)

        d_start = random.randint(0, new_d - orig_d)
        h_start = random.randint(0, new_h - orig_h)
        w_start = random.randint(0, new_w - orig_w)

        return [
            t[:, d_start:d_start + orig_d, h_start:h_start + orig_h, w_start:w_start + orig_w]
            for t in scaled
        ]

    def _apply_noise_3d(self, args):
        image_args = args[:-1]
        label = args[-1]

        std = random.uniform(0, self.noise_std)
        return [img + torch.randn_like(img) * std for img in image_args] + [label]
