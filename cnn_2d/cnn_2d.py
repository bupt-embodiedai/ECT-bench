# -*- coding: utf-8 -*-

"""
cnn_2d.py
---------
2D Convolutional Autoencoder for LBP-based Image Reconstruction.

Purpose:
--------
- This CNN-based autoencoder reconstructs higher-quality images from initial
  LBP (Local Binary Pattern) feature maps.
- Serves to validate the dataset structure and enable downstream tasks like
  material classification or shape analysis.
- Evaluates reconstruction quality via metrics such as MSE, SSIM, PSNR, ICC.

Inputs:
-------
- Single-channel grayscale images [B, 1, H, W] (H=W=100)
- Pixel values in [0,1]

Outputs:
--------
- Reconstructed images [B, 1, H, W]
- Pixel values normalized to [0,1]

Network Structure:
------------------
Encoder:
    - 4 convolutional blocks with BatchNorm, ReLU, MaxPooling
    - Feature channels increase while spatial dimensions decrease

Decoder:
    - 4 transposed convolutional blocks with BatchNorm, ReLU
    - Restores spatial dimensions to match input

Example Usage:
--------------
- import torch
- from cnn_2d import CNN2D
- model = CNN2D()
- x = torch.randn(1,1,100,100)  # dummy input
- y = model(x)
- print(x.shape, y.shape)
(1,1,100,100) (1,1,100,100)

Notes:
------
- Designed specifically for LBP-initialized grayscale images.
- Output uses Sigmoid activation to ensure pixel values remain in [0,1].
- Compatible with training/evaluation scripts such as train.py and evaluate.py.
"""

import torch
import torch.nn as nn

class CNN2D(nn.Module):
    """
    2D Convolutional Autoencoder for LBP image reconstruction.

    Args:
        None (all layers are fixed as defined)

    Forward:
        x: input tensor [B,1,H,W]
        returns: reconstructed tensor [B,1,H,W]
    """
    def __init__(self):
        super(CNN2D, self).__init__()

        # -------------------------
        # Encoder: feature extraction and downsampling
        # -------------------------
        self.encoder = nn.Sequential(
            # Conv Block 1: 100x100 -> 50x50
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 50x50 -> 25x25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3: 25x25 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 4: 12x12 -> 6x6
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # -------------------------
        # Decoder: feature restoration and upsampling
        # -------------------------
        self.decoder = nn.Sequential(
            # Deconv Block 1: 6x6 -> 12x12
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Deconv Block 2: 12x12 -> 25x25
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Deconv Block 3: 25x25 -> 50x50
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Deconv Block 4: 50x50 -> 100x100
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # restrict output to [0,1] for grayscale images
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input image tensor [B,1,H,W]

        Returns:
            torch.Tensor: Reconstructed image tensor [B,1,H,W]
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------
# Test network structure
# -------------------------
if __name__ == "__main__":
    model = CNN2D()
    print("Network architecture:")
    print(model)

    dummy_input = torch.randn(1, 1, 100, 100)  # simulate input
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
