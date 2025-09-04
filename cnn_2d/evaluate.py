# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
Evaluation script for the CNN2D autoencoder on LBP-initialized images.

Purpose:
--------
- Evaluate the trained autoencoder on the test dataset.
- Compute reconstruction metrics:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - SSIM (Structural Similarity Index)
    - PSNR (Peak Signal-to-Noise Ratio)
    - ICC (Image Correlation Coefficient)
- Output per-sample information:
    - Image filename
    - Shape category
    - Material type
- Visualize reconstructed vs. ground-truth images.
- Optionally save comparison figures to disk.

Inputs:
-------
- Pretrained model weights (.pth) located in `models_cnn2d/`
- Test dataset loaded via `dataloader.py`

Outputs:
--------
- Console logs with average evaluation metrics (MSE, RMSE, SSIM, PSNR, ICC).
- Per-image evaluation results (metrics + metadata).
- Visualization windows showing reconstructed vs. ground-truth images.
- (Optional) Saved comparison images in `results_eval/` directory.

Usage:
------
Run evaluation with default settings:
    $ python evaluate.py

Enable saving of comparison figures:
    Set `save_results = True` inside the script.

Directory Structure Example:
----------------------------
project_root/
    ├── cnn_2d.py              # Model definition
    ├── dataset.py             # Custom dataset class
    ├── dataloader.py          # Data loading logic
    ├── evaluate.py            # This script
    ├── models_cnn2d/          # Folder containing .pth model checkpoints
    ├── results_eval/          # (Optional) Generated comparison figures
    ├── imgs/                  # Input images (by shape/material)
    └── labels/                # Ground-truth label maps

Notes:
------
- Press <space> or <enter> to move to the next visualization window.
- Change `num_show_images` to adjust how many samples to visualize.
- Metrics are averaged across the entire test set before being printed.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from cnn_2d import CNN2D
from dataloader import test_loader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models_cnn2d/2d_10.pth"
num_show_images = 500
save_results = True              # Whether to save comparison figures
save_dir = "results_eval"        # Directory for saving figures
os.makedirs(save_dir, exist_ok=True)


# -----------------------
# ICC calculation
# -----------------------
def image_correlation_coefficient(original_image, reconstructed_image):
    mean_original = np.mean(original_image)
    mean_reconstructed = np.mean(reconstructed_image)
    numerator = np.sum((original_image - mean_original) * (reconstructed_image - mean_reconstructed))
    denominator = np.sqrt(
        np.sum((original_image - mean_original) ** 2) * np.sum((reconstructed_image - mean_reconstructed) ** 2)
    ) + np.finfo(float).eps
    icc = numerator / denominator
    return icc


# -----------------------
# Load model
# -----------------------
model = CNN2D().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()


# -----------------------
# Evaluation
# -----------------------
infos = []
full_list = []
labels = []
img_names = []   # image filenames
shape_names = [] # shape categories
ssim_values = []
psnr_values = []
icc_values = []

with torch.no_grad():
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    for info, label, img_name, shape_name, _ in test_loader:
        info, label = info.to(device), label.to(device)

        full = model(info)

        loss = F.mse_loss(full, label)
        total_loss += loss.item()
        rmse = torch.sqrt(loss).item()
        total_rmse += rmse

        full_list.append(full.cpu())
        labels.append(label.cpu())
        infos.append(info.cpu())

        img_names.extend(img_name)
        shape_names.extend(shape_name)

        full_np = full.cpu().numpy()
        label_np = label.cpu().numpy()

        for i in range(full_np.shape[0]):
            output_image = full_np[i].squeeze()
            label_image = label_np[i].squeeze()

            ssim_value = ssim(label_image, output_image, data_range=output_image.max() - output_image.min())
            psnr_value = psnr(label_image, output_image, data_range=output_image.max() - output_image.min())
            icc_value = image_correlation_coefficient(label_image, output_image)

            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)
            icc_values.append(icc_value)

        num_batches += 1

    average_ssim = np.mean(ssim_values)
    average_psnr = np.mean(psnr_values)
    average_icc = np.mean(icc_values)
    average_rmse = total_rmse / num_batches
    average_loss = total_loss / num_batches

    print(f"Average Loss: {average_loss:.4f}")
    print(f"Average RMSE: {average_rmse:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average PSNR: {average_psnr:.4f} dB")
    print(f"Average ICC: {average_icc:.4f}")


# -----------------------
# Visualization & Saving
# -----------------------
num_show_images = min(num_show_images, len(img_names))

full_outputs = torch.cat(full_list)
labels = torch.cat(labels)
infos = torch.cat(infos)

for i in range(num_show_images):
    test_output = full_outputs[i].cpu().squeeze().numpy()
    test_label = labels[i].cpu().squeeze().numpy()
    test_info = infos[i].cpu().squeeze().numpy()

    name = img_names[i]
    shape = shape_names[i]

    ssim_value = ssim(test_output, test_label, data_range=1)
    psnr_value = psnr(test_output, test_label, data_range=1)
    icc_value = image_correlation_coefficient(test_output, test_label)
    rmse_value = np.sqrt(((test_output - test_label) ** 2).mean())

    print(
        f"Image {i + 1} ({name}, Category: {shape}): "
        f"SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}, ICC={icc_value:.4f}, RMSE={rmse_value:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(test_output, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Output Image\n{name}\nCategory: {shape}')
    axes[0].axis('off')

    axes[1].imshow(test_label, cmap='gray', interpolation='nearest')
    axes[1].set_title('Label Image')
    axes[1].axis('off')

    plt.tight_layout()

    if save_results:
        save_path = os.path.join(save_dir, f"compare_{i+1}_{name}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison figure: {save_path}")

    plt.show(block=False)
    plt.pause(0.001)

    print("Press <space> or <enter> to continue...")
    while True:
        if plt.waitforbuttonpress():
            break

    plt.close(fig)

print("Evaluation finished.")
