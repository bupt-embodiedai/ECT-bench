# ECT Image Reconstruction Methods

This project provides a comprehensive solution for Electrical Capacitance Tomography (ECT), covering hardware design, data acquisition, and advanced image reconstruction methods.

## 🔧 Hardware and Data Collection

The `hardware-collection/` directory contains essential files for sensor design and data acquisition:
hardware-collection/
├── sensor_9e.PcbDoc # 9-electrode ECT sensor design file (Altium Designer)
└── combined.py # Data acquisition script for capacitance measurements

### Key Features:
- **Custom ECT Sensor Design** (`sensor_9e.PcbDoc`):
  - 9-electrode circular array configuration
  - Optimized shielding design for noise reduction
  - Compatible with standard PCB manufacturing processes
  - Easily modifiable for different electrode configurations

- **Data Acquisition System** (`combined.py`):
  - Interfaces with capacitance measurement hardware (e.g., AD7746)
  - Collects raw capacitance data from all electrode pairs
  - Performs real-time noise filtering and signal conditioning
  - Outputs structured CSV files for reconstruction pipelines
  - Supports both static and dynamic measurement modes

---

## 🖼️ Image Reconstruction Methods

This project contains **two complementary methods** for ECT image reconstruction:

1. **LBP-based Imaging**  
   - Generates initial images from raw ECT sensor data using **Linear Back Projection (LBP)**  
   - Produces normalized, Gaussian-smoothed 1D/2D images  
   - Can be used as input for further CNN-based reconstruction or standalone visualization

2. **CNN2D Autoencoder**  
   - Reconstructs/enhances images using a **2D Convolutional Autoencoder**  
   - Accepts **LBP-generated images** as input  
   - Can use either on-the-fly LBP processing or precomputed LBP images


## 🚀 Usage Overview

### 1. LBP Imaging

- Input: raw ECT data + background + sensitivity matrix  
- Output: normalized LBP images  
- Example:
```bash
python lbp/lbp.py
```

### 2. CNN2D Autoencoder

- Input: **LBP-generated images**  
- Output: reconstructed/enhanced ECT images  
- Can use either:
  1. **LBP run-on-the-fly**  
  2. **Precomputed LBP images** from the dataset (no need to regenerate LBP each time)

- Example:
```bash
python cnn2d/dataset_processing.py
python cnn2d/train.py
python cnn2d/evaluate.py
```

---

## 📊 Notes

- The CNN pipeline is **modular**: you can plug in other networks (U-Net, ResNet AE) with the same LBP inputs.  
- `imgs/` vs. `imgs2/` can be toggled for **contact/non-contact experiments**.  
- Precomputed LBP images **accelerate training and evaluation** while preserving reconstruction quality.

---

If you have questions or suggestions, feel free to open an issue or contact the author.
