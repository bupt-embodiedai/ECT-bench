# -*- coding: utf-8 -*-
"""
dataloader.py
-------------
Data loading utilities for the Shape–Material dataset.

Purpose:
--------
- Perform stratified splitting of the dataset into training and testing sets.
- Ensure balanced distribution: all (shape, material) combinations appear in both splits.
- Construct PyTorch DataLoaders for efficient batch iteration.

Dataset Format:
---------------
- Input images (LBP-initialized grayscale PNGs):
    imgs/<shape>/<material>_step_<n>.png
    where:
        <shape>    ∈ {circle, triangle, square}
        <material> ∈ {glass, wood, resin}
        <n>        ∈ [0, 1023]  (time step index)

- Ground truth labels (NumPy arrays):
    labels/<shape>/step_<n>.npy

Notes:
------
- The root folder name "imgs" encodes the **contact condition type**.
  To switch dataset variant:
    imgs/   -> contact dataset
    imgs2/  -> non-contact dataset
- Only need to change the dataset root path in this script.

Split Strategy:
---------------
1. Group samples by (shape, material).
2. Split each group into training and testing subsets by ratio (default: 90/10).
3. Shuffle indices globally to avoid group-ordering bias.
4. Build `Subset` datasets and wrap with DataLoader.

Returns:
--------
- `train_loader`: DataLoader for training set (shuffle enabled).
- `test_loader`:  DataLoader for testing set (no shuffle).

Example Usage:
--------------
- from dataloader import train_loader, test_loader
- for imgs, labels, img_names, shapes, materials in train_loader:
-     # imgs:       torch.Tensor, input images
-     # labels:     torch.Tensor, ground truth
-     # img_names:  list of str, filenames
-     # shapes:     list of str, shape categories
-     # materials:  list of str, material categories
-     ...
"""



import random
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
from dataset import ShapeMaterialDataset


# -------------------------
# Load full dataset
# -------------------------
dataset = ShapeMaterialDataset(imgs_root="imgs2", label_root="labels")

# -------------------------
# Group indices by (shape, material)
# -------------------------
grouped_indices = defaultdict(list)
for idx in range(len(dataset)):
    item = dataset.data_list[idx]
    shape = item["shape"]
    material = item["material"]
    grouped_indices[(shape, material)].append(idx)

# -------------------------
# Split each group into train/test
# -------------------------
train_indices, test_indices = [], []
train_ratio = 0.9
seed = 42
random.seed(seed)

for key, idx_list in grouped_indices.items():
    random.shuffle(idx_list)  # shuffle inside each group
    split_point = int(len(idx_list) * train_ratio)
    train_indices.extend(idx_list[:split_point])
    test_indices.extend(idx_list[split_point:])

# Global shuffle to avoid group ordering bias
random.shuffle(train_indices)
random.shuffle(test_indices)

# -------------------------
# Build Subsets and DataLoaders
# -------------------------
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# -------------------------
# Debug info
# -------------------------
if __name__ == "__main__":
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
