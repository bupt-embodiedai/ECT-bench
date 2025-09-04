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

---

## 📂 Project Structure
project_root/
│
│── hardware-collection/ # Sensor design and data acquisition
│ ├── sensor_9e.PcbDoc # ECT sensor PCB design (Altium)
│ └── combined.py # Data collection and processing
│
│── lbp/ # LBP reconstruction module
│ ├── lbp.py # Core LBP imaging functions
│ ├── sensitivity.py # Sensitivity matrix calculation
│ └── README.md # Detailed usage guide
│
│── cnn2d/ # Deep learning reconstruction
│ ├── dataset_processing.py # Dataset preparation
│ ├── dataset.py # PyTorch Dataset definition
│ ├── dataloader.py # Stratified DataLoader
│ ├── cnn_2d.py # Autoencoder architecture
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation metrics
│ ├── utils/ # Helper functions
│ ├── checkpoints/ # Trained model weights
│ ├── imgs/ # Contact capacitance images
│ ├── imgs2/ # Non-contact capacitance images
│ ├── labels/ # Material distribution labels
│ └── README.md # Complete usage guide
│
│── docs/ # Additional documentation
│ ├── calibration_guide.md
│ └── hardware_setup.md
│
└── README.md # Main documentation (this file)

---

## 🚀 Usage Overview

### 1. Data Collection
```bash
python hardware-collection/combined.py \
  --output data/raw_measurements.csv \
  --samples 18432 \
  --rate 12
•--output: Path to save measurements
•--samples: Number of measurements to collect
•--rate: Sampling rate in Hz
