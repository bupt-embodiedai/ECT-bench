# -*- coding: utf-8 -*-

"""
train.py
--------
Training script for the CNN2D autoencoder on LBP-initialized images.

Purpose:
--------
- Train the 2D convolutional autoencoder on the custom shape–material dataset.
- Input: LBP-initialized grayscale images.
- Output: reconstructed images that approximate ground-truth labels.
- Loss: Mean Squared Error (MSE) between predictions and labels.
- Use StepLR scheduler for gradual learning rate decay.
- Save checkpoint models periodically and the final trained model.

Inputs:
-------
- Training dataset loaded via `dataloader.py`
- Model architecture defined in `cnn_2d.py`

Outputs:
--------
- Console logs of training progress (epoch number, loss, elapsed time).
- Intermediate model checkpoints every 50 epochs, stored in `models_cnn2d/`.
- Final trained model: `models_cnn2d/2d_final.pth`

Usage:
------
Run training with default settings:
    $ python train.py

Directory Structure Example:
----------------------------
project_root/
    ├── cnn_2d.py              # Model definition
    ├── dataset.py             # Custom dataset class
    ├── dataloader.py          # Data loading logic (train/test splits)
    ├── train.py               # This script
    ├── evaluate.py            # Evaluation script
    ├── models_cnn2d/          # Folder where checkpoints will be saved
    ├── imgs/                  # Input images (by shape/material)
    └── labels/                # Ground-truth label maps

Notes:
------
- Default training runs for 500 epochs with Adam optimizer (lr=1e-4).
- Learning rate decays by factor 0.95 every 5 epochs via StepLR.
- Only image tensors and label tensors are used during training; metadata is ignored.
- Adjust `num_epochs`, `learning_rate`, and `batch_size` (in `dataloader.py`) to fit hardware resources.
"""


import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from cnn_2d import CNN2D
from dataloader import train_loader  # Custom dataset loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CNN2D().to(device)

# Training hyperparameters
num_epochs = 100
learning_rate = 1e-4

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

# Set model to training mode
model.train()

# Directory to save checkpoints
checkpoint_dir = "models_cnn2d"
os.makedirs(checkpoint_dir, exist_ok=True)

# -------------------------
# Training loop
# -------------------------
for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = 0.0

    for info, label, _, _, _ in train_loader:  # During the training phase, only images and labels are used
        info, label = info.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(info)

        loss = F.mse_loss(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    end_time = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {end_time-start_time:.2f}s")

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"2d_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# Save final model
final_model_path = os.path.join(checkpoint_dir, "2d_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Training complete! Final model saved at: {final_model_path}")
