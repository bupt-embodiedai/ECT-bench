# -*- coding: utf-8 -*-

"""
lbp.py
------
LBP-based Imaging for ECT Sensor Data

Purpose:
--------
- Compute LBP (Linear Back Projection) images from raw ECT capacitance measurements.
- Apply Gaussian smoothing and normalization for CNN autoencoder input.
- Support batch loading of raw data, background, and sensitivity matrix from CSV files.
- Sensitivity matrix (S_matrix) maps sensor readings to pixels and can be obtained from simulation
  or used with real ECT measurements.

Usage:
------
1. Load CSV files containing raw ECT measurements, background, and sensitivity matrix.
2. Compute normalized LBP images using `lbp_imaging()`.
3. Optionally reshape 1D images to 2D for visualization or CNN input.

Example:
--------
```python
from lbp import lbp_imaging, load_ect_csv
import matplotlib.pyplot as plt
import numpy as np

raw_data, background_avg, S_matrix = load_ect_csv("TestData/0_0.csv",
                                                  "TestData/bkg.csv",
                                                  "TestData/S_matrix.csv")

# Compute 1D LBP image
image_1d = lbp_imaging(raw_data, background_avg, S_matrix)

# Reshape for 2D visualization
grid_size = int(np.sqrt(image_1d.shape[0]))
image_2d = image_1d.reshape((grid_size, grid_size))

plt.imshow(image_2d, cmap='gray_r', origin='lower')
plt.colorbar(label='Normalized Value')
plt.title("LBP Imaging 2D")
plt.show()
"""


import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt

def lbp_imaging(raw_data, background_avg, S_matrix, smoothing_sigma=0.7):
    """
    Perform LBP-based imaging from ECT sensor data.

    Parameters
    ----------
    raw_data : np.ndarray
        Current raw capacitance values, shape (num_sensors,)
    background_avg : np.ndarray
        Background average values, shape (num_sensors,)
    S_matrix : np.ndarray
        Sensitivity matrix, shape (num_sensors, num_pixels)
    smoothing_sigma : float
        Gaussian smoothing sigma (default=0.7)

    Returns
    -------
    image_normalized : np.ndarray
        Normalized 1D image vector, shape (num_pixels,)
    """
    epsilon = 1e-10
    diff = raw_data - background_avg

    denom = background_avg.copy()
    denom[denom < epsilon] = epsilon

    c_norm = diff / denom

    # LBP imaging
    image = c_norm @ S_matrix

    # Gaussian smoothing
    if smoothing_sigma > 0:
        image = gaussian_filter(image, sigma=smoothing_sigma)

    # Normalize to 0~1
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    image_normalized = (image - p2) / (p98 - p2 + epsilon)
    image_normalized = np.clip(image_normalized, 0, 1)

    return image_normalized


def load_ect_csv(raw_csv_path, bkg_csv_path, S_csv_path):
    """
    Load raw_data, background_avg, and sensitivity matrix from CSV files.
    """
    raw_data = pd.read_csv(raw_csv_path, header=None).iloc[0, 3:].to_numpy(dtype=float)
    background_avg = pd.read_csv(bkg_csv_path, header=None).iloc[0, 3:].to_numpy(dtype=float)
    S_matrix = pd.read_csv(S_csv_path, header=None).iloc[:, 3:].to_numpy(dtype=float).T
    return raw_data, background_avg, S_matrix


if __name__ == "__main__":
    raw_csv_path = "TestData/0_0.csv"
    bkg_csv_path = "TestData/bkg.csv"
    S_csv_path = "TestData/S_matrix.csv"

    raw_data, background_avg, S_matrix = load_ect_csv(raw_csv_path, bkg_csv_path, S_csv_path)

    image = lbp_imaging(raw_data, background_avg, S_matrix)

    num_pixels = image.shape[0]
    grid_size = int(np.sqrt(num_pixels))
    image_2d = image.reshape((grid_size, grid_size))

    plt.figure(figsize=(6,6))
    plt.imshow(image_2d, cmap='gray_r', origin='lower')
    plt.colorbar(label='Normalized Value')
    plt.title("LBP Imaging 2D")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.show()
