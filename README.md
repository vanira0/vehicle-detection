# Vehicle Damage Detection Pipeline

A modular, production-ready vehicle damage detection system using a 3-stage pipeline architecture. Built with PyTorch and designed for **strategy swappability** — experiment with different model architectures, training techniques, and hyperparameters without changing any pipeline code.

## Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────┐
│ Stage 0: Vehicle Isolation  │  YOLO11n-seg (single class: vehicle)
│ "Isolate the target vehicle"│  Outputs bbox + pixel mask
└─────────────┬───────────────┘
              │ (Masked vehicle image)
              ▼
┌─────────────────────────────┐
│ Stage 1: Gatekeeper         │  Binary CNN (ResNet50 / MobileNetV3)
│ "Is this a damaged car?"    │  Filters out irrelevant images early
└─────────────┬───────────────┘
              │ (Damaged)
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────────┐  ┌──────────────┐
│ Stage 2:     │  │ Stage 3:     │
│ Damage Seg.  │  │ Part Seg.    │  Mask R-CNN / YOLOv8 / Swin
│ (dent, crack)│  │ (hood, door) │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
┌─────────────────────────────┐
│ Orchestrator (IoU Overlap)  │  Maps damage → part
│ "Severe Dent on Left Fender"│
└─────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train the baseline Mask R-CNN for damage detection
python scripts/train.py --config configs/damage/maskrcnn_resnet50.yaml

# Try a different strategy (Swin Transformer backbone)
python scripts/train.py --config configs/damage/maskrcnn_swin.yaml

# Override hyperparameters via CLI
python scripts/train.py --config configs/damage/maskrcnn_resnet50.yaml \
    --set training.optimizer.lr=0.001 training.epochs=30
```

### 3. Compare Experiments

```bash
python scripts/compare.py --runs damage_maskrcnn_resnet50_v1 damage_maskrcnn_swin_v1
```

### 4. Run Inference

```bash
python scripts/infer.py \
    --image test_car.jpg \
    --gatekeeper runs/gatekeeper_v1/checkpoints/best.pth \
    --damage runs/damage_v1/checkpoints/best.pth \
    --parts runs/parts_v1/checkpoints/best.pth
```

### 5. Export for Deployment

```bash
python scripts/export.py --checkpoint runs/damage_v1/checkpoints/best.pth --format onnx --output model.onnx
```

## How to Add a New Model Strategy

1. **Create the model file** — e.g., `src/models/damage/my_new_model.py`
2. **Implement the interface** — inherit from `BaseDetector` (or `BaseClassifier`)
3. **Register it** — add `@register_model("my_new_model")` decorator
4. **Create a config** — `configs/damage/my_new_model.yaml`
5. **Train** — `python scripts/train.py --config configs/damage/my_new_model.yaml`

No other code changes needed. The Trainer, data pipeline, and evaluation system work automatically.

## Project Structure

```
vehicle-detection/
├── configs/              # YAML experiment configs with inheritance
│   ├── base.yaml         # Shared defaults
│   ├── gatekeeper/       # Gatekeeper model configs
│   ├── damage/           # Damage segmentation configs
│   └── parts/            # Part segmentation configs
├── src/
│   ├── data/             # Datasets, augmentations, splitting
│   ├── models/           # Model registry + implementations
│   ├── training/         # Model-agnostic trainer, callbacks
│   ├── evaluation/       # Metrics, evaluator, comparator
│   ├── inference/        # Pipeline, orchestrator, exporters
│   └── utils/            # Config, logging, seed, visualization
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── runs/                 # Experiment outputs (gitignored)
└── data/                 # Dataset root (gitignored)
```

## Available Model Strategies

| Name | Stage | Type | Architecture |
|------|-------|------|-------------|
| `yolo11_vehicle_seg` | Vehicle | Instance Seg. | YOLO11n-seg |
| `resnet50_classifier` | Gatekeeper | Classification | ResNet50 + FC head |
| `mobilenetv3_classifier` | Gatekeeper | Classification | MobileNetV3 + FC head |
| `maskrcnn_resnet50` | Damage | Instance Seg. | Mask R-CNN (ResNet50-FPN) |
| `maskrcnn_swin` | Damage | Instance Seg. | Mask R-CNN (Swin-T backbone) |
| `yolov8_seg` | Damage | Instance Seg. | YOLOv8 segmentation |
| `maskrcnn_resnet50_parts` | Parts | Instance Seg. | Mask R-CNN (ResNet50-FPN) |

## Running Tests

```bash
pytest tests/ -v
```
