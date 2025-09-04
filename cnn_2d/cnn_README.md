# CNN2D Autoencoder for ECT-MAT

This project implements a **2D Convolutional Autoencoder** for image reconstruction using a custom **Electrical Capacitance Tomography (ECT) dataset**.  
It includes dataset preparation, dataloading, network architecture, training, and evaluation scripts.

---

## 📂 Project Structure

```
project_root/
│── dataset_processing.py    # Prepare dataset (imgs/, imgs2/, labels/)
│── dataset.py               # PyTorch Dataset definition
│── dataloader.py            # Stratified DataLoader split (train/test)
│── cnn_2d.py                # CNN2D autoencoder network
│── train.py                 # Training script
│── evaluate.py              # Evaluation script
│── checkpoints/             # Saved model weights (after training)
│── imgs/                    # Processed contact images
│   ├── circle/
│   │    ├── glass_step_0.png
│   │    ├── wood_step_1.png
│   │    └── resin_step_2.png
│   ├── square/
│   └── triangle/
│── imgs2/                   # Processed non-contact images (same structure as imgs/)
│── labels/                  # Shared labels by shape
│   ├── circle/
│   │    ├── step_0.npy
│   │    ├── step_1.npy
│   │    └── ...
│   ├── square/
│   └── triangle/
└── README.md                # Documentation (this file)
```

---

## 🗂 Dataset

### Input Images
- Located in `imgs/` (contact) and `imgs2/` (non-contact).  
- Naming convention:  
  ```
  imgs/<shape>/<material>_step_<n>.png
  ```
  where:
  - `<shape>` ∈ {circle, square, triangle}  
  - `<material>` ∈ {glass, wood, resin}  
  - `<n>` ∈ [0, 1023] (time step index)

### Ground-Truth Labels
- Located in `labels/<shape>/`
- Naming convention:
  ```
  labels/<shape>/step_<n>.npy
  ```

**Note:**  
Labels are **shared across materials and contact conditions** (reducing storage and I/O).  

---

## ⚙️ Components

### 1. `dataset_processing.py`
- Prepares the dataset by restructuring raw ECT data.  
- Splits **contact** and **non-contact** images into `imgs/` and `imgs2/`.  
- Consolidates labels into `labels/` shared across all materials.  
- Renames files to ensure uniform naming:
  - Images: `<material>_step_<num>.png`  
  - Labels: `step_<num>.npy`

---

### 2. `dataset.py`
- Defines `ShapeMaterialDataset`, a PyTorch `Dataset`.  
- Returns:
  ```python
  img, label, img_name, shape, material
  ```
- Loads grayscale images and corresponding labels.  

---

### 3. `dataloader.py`
- Performs **stratified splitting** by `(shape, material)` to ensure balanced train/test sets.  
- Builds PyTorch `DataLoader`:
  ```python
  train_loader, test_loader
  ```

---

### 4. `cnn_2d.py`
- Implements the **2D Convolutional Autoencoder**.  
- Encoder: 4 convolutional blocks with downsampling.  
- Decoder: 4 transposed convolutional blocks with upsampling.  
- Input: `[B,1,100,100]` grayscale images.  
- Output: `[B,1,100,100]` reconstructed images.  

---

### 5. `train.py`
- Trains the CNN2D autoencoder on the dataset.  
- Features:
  - Loss: **Mean Squared Error (MSE)**
  - Optimizer: **Adam**
  - Scheduler: **StepLR** (learning rate decay)
  - Saves periodic checkpoints and final model
- Example:
  ```bash
  python train.py
  ```

---

### 6. `evaluate.py`
- Evaluates a trained model on the test set.  
- Computes reconstruction metrics:
  - **MSE** (Mean Squared Error)  
  - **RMSE** (Root Mean Squared Error)  
  - **SSIM** (Structural Similarity Index)  
  - **PSNR** (Peak Signal-to-Noise Ratio)  
  - **ICC** (Image Correlation Coefficient)  
- Outputs:
  - Image name, shape, material
  - Visualization: reconstructed vs. ground-truth

---

## 🚀 Usage

### Step 1: Prepare dataset
```bash
python dataset_processing.py
```

### Step 2: Train autoencoder
```bash
python train.py
```

### Step 3: Evaluate model
```bash
python evaluate.py
```

---

## 📊 Example Results

- After training, the autoencoder is able to **reconstruct ECT images** with high similarity.  
- Metrics (SSIM/PSNR/ICC) confirm that the dataset structure is meaningful and suitable for further tasks such as:
  - Shape recognition
  - Material classification
  - Advanced ECT inversion methods

---

## 📝 Notes
- Change dataset paths in `dataset_processing.py` before running.  
- `imgs/` vs. `imgs2/` can be toggled in `dataloader.py` for **contact/non-contact experiments**.  
- The project is modular: you can plug in different networks (e.g., U-Net, ResNet AE) using the same dataset/dataloader pipeline.  

--- 

If you have questions or suggestions, feel free to open an issue or contact the author.

