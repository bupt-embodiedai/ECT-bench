# -*- coding: utf-8 -*-

"""
dataset_processing.py
---------------------
Data preparation script for the custom ECT dataset, organizing images and labels
into a unified structure for 2D CNN-based autoencoder training.

Purpose:
--------
- Transform the original raw ECT dataset into a structured format suitable for
  convolutional autoencoder training and evaluation.
- Separate contact and non-contact images into `imgs/` and `imgs2/`.
- Consolidate labels by shape into a shared `labels/` folder (independent of
  contact condition and material).
- Standardize file naming:
    * Images → <material>_step_<num>.png
    * Labels → step_<num>.npy
- Reduce disk usage and loading overhead by sharing labels across variants.

Inputs:
-------
- Original dataset directory with the following structure:
    ECTDATA/
        contact/ or non_contact/
            <material>/       # glass / resin / wood
                <shape>/      # circle / square / triangle
                    lbp_reconstruction/  # raw images (001.png, 002.png, ...)
                    label/                # ground-truth labels (0.npy, 1.npy, ...)

Outputs:
--------
- Reorganized dataset in the current working directory:
    imgs/         # contact images
        <shape>/
            <material>_step_<num>.png
    imgs2/        # non-contact images
        <shape>/
            <material>_step_<num>.png
    labels/       # shared labels by shape
        <shape>/
            step_0.npy
            step_1.npy
            ...

Console logs report:
- Number of processed image files
- Number of shapes with extracted labels

Usage:
------
Run the script directly from terminal:
    $ python dataset_processing.py

Example directory structure after processing:
---------------------------------------------
current_working_dir/
    ├── imgs/        # contact images
    ├── imgs2/       # non-contact images
    └── labels/      # shape-based shared labels

Notes:
------
- Images are renamed by removing leading zeros and prepending material name.
- Labels are shared across contact conditions and materials (organized only by shape).
- The output dataset is tailored for 2D CNN autoencoder tasks:
    - Input images from imgs/ or imgs2/
    - Ground-truth labels from labels/
- Ensure the original dataset strictly follows the expected folder hierarchy
  (contact/non_contact → material → shape → lbp_reconstruction/label).
"""


import os
import shutil

# -------------------- Configuration --------------------
base_dir = r"E:\ECT\ECTgit\ECTDATA"   # Path to the original dataset
new_base_dir = os.getcwd()            # Current working directory as output root

# Mapping for contact type to output folders
# "contact" → imgs, "non_contact" → imgs2
contact_map = {"contact": "imgs", "non_contact": "imgs2"}

# -------------------- Image Processing --------------------
# Processing log list
log_list = []

# Walk through all subdirectories and files in the dataset
for root, dirs, files in os.walk(base_dir):
    for file_name in files:
        # Only process PNG files
        if not file_name.lower().endswith(".png"):
            continue

        # Determine image type: contact or non_contact
        if "non_contact" in root:
            contact_type = "non_contact"
            contact_index = root.index("non_contact")
        elif "contact" in root:
            contact_type = "contact"
            contact_index = root.index("contact")
        else:
            # Skip if neither contact nor non_contact appears in the path
            continue

        # Extract material and shape from the path after contact/non_contact
        sub_path = root[contact_index:].split(os.sep)
        if len(sub_path) < 4:
            # Path too short, possibly missing material or shape, skip
            continue

        material = sub_path[1]  # Material (glass/resin/wood)
        shape = sub_path[2]     # Shape (circle/square/triangle)

        # Create output directory imgs/<shape> or imgs2/<shape>
        new_folder = os.path.join(new_base_dir, contact_map[contact_type], shape)
        os.makedirs(new_folder, exist_ok=True)

        # Standardize file name as <material>_step_<num>.png (remove leading zeros)
        try:
            num = int(os.path.splitext(file_name)[0])
        except ValueError:
            # Skip files whose names cannot be converted to integers
            continue
        new_file_name = f"{material}_step_{num}.png"

        # Copy image to the new directory
        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(new_folder, new_file_name)
        shutil.copy(src_file, dst_file)

        # Record log
        log_list.append(f"{src_file} → {dst_file}")

# Print total processed image count
print(f"Image processing complete, total {len(log_list)} files processed.")

# -------------------- Label Extraction --------------------
# Label output directory labels/
label_new_base = os.path.join(new_base_dir, "labels")
os.makedirs(label_new_base, exist_ok=True)

# Set of processed shapes to avoid duplication
processed_shapes = set()

# Labels are shared between contact and non_contact, so only traverse contact/
contact_dir = os.path.join(base_dir, "contact")

for material in os.listdir(contact_dir):
    material_path = os.path.join(contact_dir, material)
    if not os.path.isdir(material_path):
        continue

    # Traverse each shape folder
    for shape in os.listdir(material_path):
        shape_path = os.path.join(material_path, shape)
        label_path = os.path.join(shape_path, "label")
        if not os.path.isdir(label_path):
            continue

        if shape in processed_shapes:
            # Already extracted labels for this shape, skip
            continue

        # Create output folder labels/<shape>
        new_label_dir = os.path.join(label_new_base, shape)
        os.makedirs(new_label_dir, exist_ok=True)

        # Copy all .npy files (0-1023) under label/
        label_files = sorted([f for f in os.listdir(label_path) if f.endswith(".npy")])
        for f in label_files:
            src_label = os.path.join(label_path, f)
            dst_label = os.path.join(new_label_dir, f"step_{int(os.path.splitext(f)[0])}.npy")
            shutil.copy(src_label, dst_label)
            # Debug log (commented out by default)
            # print(f"Copied label: {src_label} → {dst_label}")

        # Mark this shape as processed
        processed_shapes.add(shape)

# Print total shapes processed for labels
print(f"Total {len(processed_shapes)} shapes of labels extracted.")

