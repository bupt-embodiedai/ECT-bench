# -*- coding: utf-8 -*-

"""
dataset.py
----------
Custom PyTorch Dataset class for the shape–material image dataset.

Purpose:
--------
- Define `ShapeMaterialDataset` for paired image–label loading.
- Support both contact (`imgs/`) and non-contact (`imgs2/`) datasets.
- Handle multiple shapes (circle, triangle, square) and materials (glass, wood, resin).
- Normalize ground-truth label arrays to [0, 1].
- Provide metadata (filename, shape, material) along with tensors.

Dataset Description:
--------------------
- Input images:
    - LBP-initialized grayscale `.png` files
    - Stored in: `imgs/<shape>/<material>_step_<n>.png`
      where:
        <shape> ∈ {circle, triangle, square}
        <material> ∈ {glass, wood, resin}
        <n> ∈ [0, 1023] (time step index)

- Ground-truth labels:
    - NumPy `.npy` arrays
    - Stored in: `labels/<shape>/step_<n>.npy`

- Two dataset variants:
    - Contact dataset:     `imgs/`
    - Non-contact dataset: `imgs2/`
    - Switching datasets requires only changing the `imgs_root` argument.

Inputs:
-------
- imgs_root (str):   Root directory for input `.png` images.
- label_root (str):  Root directory for ground-truth `.npy` labels.
- transform (callable, optional): Image transform (default: `torchvision.transforms.ToTensor()`).

Outputs (per sample):
---------------------
- img_tensor:    torch.Tensor, shape [1, H, W], grayscale input image.
- label_tensor:  torch.Tensor, shape [1, H, W], normalized ground-truth label.
- img_name:      str, image filename.
- shape_name:    str, one of ["circle", "triangle", "square"].
- material_name: str, one of ["glass", "wood", "resin"].

Example Directory Structure:
----------------------------
project_root/
    ├── dataset.py
    ├── dataloader.py
    ├── imgs/                   # Contact dataset
    │   ├── circle/
    │   │   ├── glass_step_0.png
    │   │   ├── wood_step_1.png
    │   │   └── resin_step_2.png
    │   └── square/...
    ├── imgs2/                  # Non-contact dataset (same structure as imgs/)
    └── labels/
        ├── circle/
        │   ├── step_0.npy
        │   ├── step_1.npy
        │   └── step_2.npy
        └── square/...

Usage:
------
- from dataset import ShapeMaterialDataset
- dataset = ShapeMaterialDataset(imgs_root="imgs", label_root="labels")
- img, label, img_name, shape, material = dataset[0]

Notes:
------
- Each label is min–max normalized independently.
- The dataset will skip samples if the corresponding `.npy` label file is missing.
- Metadata is preserved for evaluation, logging, and visualization.
"""


import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShapeMaterialDataset(Dataset):
    """
    PyTorch Dataset for shape–material images.

    Args:
        imgs_root (str): Root directory of input images.
        label_root (str): Root directory of label .npy files.
        transform (callable, optional): Transform to apply to images
                                        (default: transforms.ToTensor()).
    """
    def __init__(self, imgs_root, label_root, transform=None):
        self.imgs_root = imgs_root
        self.label_root = label_root
        self.transform = transform if transform else transforms.ToTensor()

        self.data_list = []
        shapes = ['circle', 'triangle', 'square']

        for shape in shapes:
            img_folder = os.path.join(imgs_root, shape)
            label_folder = os.path.join(label_root, shape)

            if not os.path.isdir(img_folder) or not os.path.isdir(label_folder):
                print(f"Missing folder: {img_folder} or {label_folder}")
                continue

            for fname in os.listdir(img_folder):
                if not fname.endswith('.png'):
                    continue

                # Extract step index from filename
                match = re.search(r'step_\d+', fname)
                if not match:
                    print(f"Filename format unexpected: {fname}")
                    continue

                step_name = match.group()
                label_name = step_name + '.npy'
                label_path = os.path.join(label_folder, label_name)

                if not os.path.exists(label_path):
                    print(f"Label file missing: {label_path}")
                    continue

                material = fname.split('_')[0]  # e.g., "glass", "wood", "resin"

                self.data_list.append({
                    'img_path': os.path.join(img_folder, fname),
                    'label_path': label_path,
                    'shape': shape,
                    'material': material
                })

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Load and return a sample."""
        item = self.data_list[idx]

        # Load grayscale image
        img = Image.open(item['img_path']).convert('L')
        img_tensor = self.transform(img)

        # Load and normalize label
        label = np.load(item['label_path']).astype(np.float32)
        label = (label - label.min()) / (label.max() - label.min() + 1e-8)
        label_tensor = torch.tensor(label).unsqueeze(0)

        # Metadata
        img_name = os.path.basename(item['img_path'])
        shape_name = item['shape']
        material_name = item['material']

        return img_tensor, label_tensor, img_name, shape_name, material_name


# Quick test
if __name__ == "__main__":
    dataset = ShapeMaterialDataset(imgs_root="imgs", label_root="labels")
    print(f"Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print("Sample contents:", [type(x) for x in sample])
