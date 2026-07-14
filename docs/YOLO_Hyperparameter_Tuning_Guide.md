# YOLO Hyperparameter Tuning Guide

This guide explains how to configure and pass training hyperparameters for Ultralytics YOLO models (e.g., YOLO11, YOLOv8, YOLO26) natively through our custom configuration YAML files.

## Overview

All YOLO wrappers in this project natively pass arguments to Ultralytics. The system will automatically map the standard fields in the `training` section of your config YAML to the YOLO engine. For fine-grained control over advanced YOLO settings (augmentations, loss gains, specific optimization tweaks), you can use the `yolo_kwargs` block under the `training` section.

## Configuration YAML Structure

Here is an example configuration showcasing how to tune hyperparameters:

```yaml
_base_: ../base.yaml

experiment_name: yolo11_damage_seg_v1

model:
  name: yolo11_seg       # or yolo26_seg, yolov8_seg, yolo11n_cls_angle_classifier
  stage: damage
  weights: yolo11s-seg.pt

data:
  yaml_path: data/car_DD_damage_yolo/data.yaml
  image_size: 1024

training:
  # --- Standard Parameters ---
  epochs: 100                 # Total training epochs
  batch_size: 16              # Batch size (set to -1 for auto-batching)
  optimizer:
    lr: 0.01                  # Initial learning rate (maps to lr0)
    weight_decay: 0.0005      # L2 penalty

  # --- Advanced YOLO Hyperparameters ---
  # Every key-value pair here is passed directly to the YOLO model.train() function.
  yolo_kwargs:
    optimizer: SGD            # Optimizer choice (e.g., SGD, Adam, AdamW, auto)
    momentum: 0.937           # Momentum factor (SGD) or beta1 (Adam)
    patience: 5               # Early stopping patience (epochs to wait)
    cos_lr: true              # Use a cosine learning rate scheduler

    # -- Loss Gains --
    box: 7.5                  # Box loss gain weight
    cls: 0.5                  # Classification loss gain weight
    dfl: 1.5                  # Distribution Focal Loss weight
    label_smoothing: 0.0      # Label smoothing epsilon (0.0 to 0.1)

    # -- Augmentations --
    hsv_h: 0.015              # Image HSV-Hue augmentation (fraction)
    hsv_s: 0.7                # Image HSV-Saturation augmentation (fraction)
    hsv_v: 0.4                # Image HSV-Value augmentation (fraction)
    degrees: 0.0              # Image rotation (+/- degrees)
    translate: 0.1            # Image translation (+/- fraction)
    scale: 0.5                # Image scaling (+/- gain)
    shear: 0.0                # Image shear (+/- degrees)
    perspective: 0.0          # Image perspective (+/- fraction)
    flipud: 0.0               # Image flip up-down (probability)
    fliplr: 0.5               # Image flip left-right (probability)
    mosaic: 1.0               # Image mosaic (probability)
    mixup: 0.0                # Image mixup (probability)
    copy_paste: 0.0           # Segment copy-paste (probability)

output:
  runs_dir: runs/damage_yolo11
```

## Key Hyperparameters

### Optimization & Training Duration
*   **`epochs`**: Total number of passes over the dataset.
*   **`batch_size`**: Number of images per batch. Set to `-1` for YOLO to automatically calculate the maximum batch size for your GPU memory.
*   **`imgsz` / `image_size`**: Target image size for training.
*   **`yolo_kwargs.optimizer`**: Optimizer selection. Typically `auto`, `SGD`, `Adam`, or `AdamW`.
*   **`lr` / `yolo_kwargs.lr0`**: Initial learning rate.
*   **`yolo_kwargs.patience`**: Number of epochs to wait without observable improvement before stopping training early.

### Augmentations (Crucial for Damage & Parts Detection)
The augmentation pipeline is very robust in YOLO models. Tweaking these parameters helps your model generalize:
*   **`hsv_h`, `hsv_s`, `hsv_v`**: Modifies the hue, saturation, and brightness. This is highly recommended for vehicles as lighting conditions often vary (e.g., reflections, night time).
*   **`mosaic`**: Combines 4 training images into one. It allows the model to learn context and helps find smaller objects. Default is usually `1.0` (100% chance).
*   **`copy_paste`**: Highly beneficial for rare classes (like specific car damages). Copies a segmented mask and pastes it onto another image to synthetically generate more training data.

### Loss Adjustments
If your model is struggling to correctly bound objects or classify damage accurately:
*   **`box`**: Increase this if bounding boxes or masks are too loose or inaccurate.
*   **`cls`**: Increase this if the model is confusing different damage classes (e.g., confusing "scratch" with "crack").

## How It Works Under The Hood

The generic training pipeline defined in our source code (`src/models/yolo_segmentation.py`, `src/models/damage/yolov8_seg.py`, `src/models/angle/yolo_classifier.py`) extracts any parameters specified inside the `yolo_kwargs` dictionary. It merges them with standard hyperparameters (like `epochs`, `batch`, `imgsz`) and passes the unified dictionary natively into the `Ultralytics` engine.

This ensures that any newly introduced parameter in future YOLO versions can be instantly utilized by just updating your configuration YAML file, without touching the Python source code.
