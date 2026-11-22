"""Data loading and augmentation for CIFAR-10.

Augmentation pipeline for the improved baseline:
  - Random crop + horizontal flip  (standard)
  - RandAugment                    (colour / geometric variety)
  - Mixup / CutMix                 (applied at the batch level in the trainer)
"""

import pickle
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_cifar10(data_dir: str):
    """Return (x_train, y_train, x_test, y_test) as float32 arrays.
    x shape: [N, 3, 32, 32], values in [0, 1].
    """
    def _unpickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    xs, ys = [], []
    for i in range(1, 6):
        d = _unpickle(f"{data_dir}/data_batch_{i}")
        xs.append(d[b'data'])
        ys.extend(d[b'labels'])
    x_train = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(ys, dtype=np.int64)

    d = _unpickle(f"{data_dir}/test_batch")
    x_test = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test  = np.array(d[b'labels'], dtype=np.int64)

    return x_train, y_train, x_test, y_test


class CIFAR10Dataset(Dataset):
    """Dataset with optional augmentation.

    Args:
        x: float32 array [N, 3, 32, 32] in [0,1]
        y: int64 array  [N]
        training: whether to apply augmentation
        alternating_flip: deterministic 50% flip instead of random
        use_randaugment: if True, apply RandAugment after crop/flip
        index_offset: starting sample index used for alternating flip parity
    """

    # RandAugment: built from torchvision, shared across instances
    _randaugment = T.RandAugment(num_ops=2, magnitude=9)

    def __init__(self, x, y, training=False, alternating_flip=False,
                 use_randaugment=False, index_offset=0):
        self.x = x
        self.y = y
        self.training = training
        self.alternating_flip = alternating_flip
        self.use_randaugment = use_randaugment
        self.index_offset = index_offset

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx].copy()  # [3, 32, 32], float32 in [0,1]
        label = self.y[idx]

        if self.training:
            # Random crop: pad 4 on each side, crop back to 32×32
            padded = np.pad(img, ((0,0),(4,4),(4,4)), mode='reflect')
            i = np.random.randint(0, 9)
            j = np.random.randint(0, 9)
            img = padded[:, i:i+32, j:j+32]

            # Horizontal flip
            if self.alternating_flip:
                if (idx + self.index_offset) % 2 == 1:
                    img = img[:, :, ::-1].copy()
            else:
                if np.random.rand() > 0.5:
                    img = img[:, :, ::-1].copy()

            if self.use_randaugment:
                # RandAugment expects uint8 [C,H,W] tensor
                img_uint8 = torch.from_numpy(
                    (img.clip(0, 1) * 255).astype(np.uint8))
                img_uint8 = self._randaugment(img_uint8)
                img = img_uint8.float().numpy() / 255.0

        # Normalise with dataset statistics
        img = (img - CIFAR10_MEAN[:, None, None]) / CIFAR10_STD[:, None, None]
        return torch.from_numpy(img.copy()), torch.tensor(label, dtype=torch.long)


def make_loaders(x_train, y_train, x_test, y_test,
                 batch_size=128, num_workers=4,
                 alternating_flip=False, train_ratio=0.9,
                 use_randaugment=False):
    """Split train into train/val and return DataLoaders."""
    n = len(y_train)
    split = int(n * train_ratio)
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_val, y_val = x_train[split:], y_train[split:]

    ds_train = CIFAR10Dataset(x_tr, y_tr, training=True,
                              alternating_flip=alternating_flip,
                              use_randaugment=use_randaugment)
    ds_val   = CIFAR10Dataset(x_val, y_val, training=False)
    ds_test  = CIFAR10Dataset(x_test, y_test, training=False)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
