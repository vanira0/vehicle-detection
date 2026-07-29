# Vehicle Damage Detection System - Implementation Plan

## Project Goal

Build an industrial-grade, lightweight Vehicle Damage Detection System for insurance claim automation capable of:

- Detecting vehicle damage accurately
- Identifying damaged vehicle parts
- Eliminating false positives from background objects and reflections
- Estimating damage severity
- Mapping damage to repair cost
- Deploying efficiently on cloud/edge infrastructure

Target Metrics:

- mAP@0.5:0.95 > 90%
- Precision > 95%
- False Positive Rate < 3%
- Inference < 1 second (GPU)
- Lightweight deployment (YOLO11 Nano/Small)

---

# Overall System Architecture

```
                 Input Image
                      │
                      ▼
        Image Quality Assessment
                      │
        Reject blurry/dark images
                      │
                      ▼
          Vehicle Detection (YOLO11n)
                      │
                      ▼
      Vehicle Segmentation (YOLO11n-Seg)
                      │
      Crop only primary vehicle region
                      │
                      ▼
      Vehicle Orientation Classification
                      │
                      ▼
      Vehicle Part Segmentation (YOLO11s-Seg)
                      │
                      ▼
        Damage Segmentation (YOLO11s-Seg)
                      │
                      ▼
       Damage-Part IoU Mapping Engine
                      │
                      ▼
       Rule-based Validation Engine
                      │
                      ▼
        Severity Classification
                      │
                      ▼
      Damage Area Calculation
                      │
                      ▼
        Repair Cost Prediction
                      │
                      ▼
      Insurance Claim Report
```

---

# Why Modular Architecture?

Instead of training one large model, every model learns only one task.

Advantages

- Better accuracy
- Easier debugging
- Easier retraining
- Lightweight deployment
- Lower false positives
- Better explainability

---

# Model Breakdown

---

## Model 1 — Image Quality Assessment

Purpose

Reject unusable images before inference.

Architecture

MobileNetV4

Input

Entire Image

Output Classes

- Good
- Blurry
- Dark
- Overexposed
- Cropped Vehicle
- Multiple Vehicles
- Reflection Dominated
- Occluded

If image quality is poor

Reject inference.

---

## Model 2 — Vehicle Detection

Purpose

Locate the primary vehicle.

Architecture

YOLO11 Nano

Classes

```
vehicle
```

Annotation

Bounding Boxes

Dataset

20,000+ images

---

## Model 3 — Vehicle Segmentation

Purpose

Remove background completely.

Architecture

YOLO11 Nano Segmentation

Classes

```
vehicle
```

Annotation

Polygon around entire vehicle.

Important

Annotate

- Entire visible vehicle

Never include

- Shadow
- Ground
- Reflection
- Background

---

## Model 4 — Vehicle Orientation

Purpose

Determine camera angle.

Classes

```
Front

Rear

Left

Right

Front Left

Front Right

Rear Left

Rear Right

Top
```

Useful for

- Rule engine
- Report generation

---

## Model 5 — Vehicle Part Segmentation

Purpose

Identify every visible vehicle component.

Architecture

YOLO11 Small Segmentation

Annotation Type

Polygon

IMPORTANT

Annotate ALL visible vehicle parts.

Not only damaged parts.

Example

Image

Visible

✓ Hood

✓ Front Bumper

✓ Left Fender

✓ Left Door (Damaged)

✓ Windshield

✓ Mirror

✓ Wheel

Everything visible gets annotated.

Never annotate invisible parts.

---

Vehicle Part Classes

```
hood

front_bumper

rear_bumper

left_front_fender

right_front_fender

left_rear_fender

right_rear_fender

left_front_door

right_front_door

left_rear_door

right_rear_door

roof

trunk

windshield

rear_glass

left_window

right_window

left_headlight

right_headlight

left_tail_light

right_tail_light

left_mirror

right_mirror

grille

license_plate

wheel

tire
```

---

## Model 6 — Damage Segmentation

Purpose

Locate damage precisely.

Architecture

YOLO11 Small Segmentation

Classes

```
dent

scratch

broken

missing

shatter
```

Annotation Type

Polygon

Important Rules

Never annotate entire panel.

Annotate only damaged pixels.

Wrong

Entire Door Polygon

Correct

Small Dent Polygon

For scratches

Draw thin polygon following scratch.

If scratch overlaps dent

Annotate separately.

---

## Model 7 — Severity Classification

Purpose

Estimate repair severity.

Input

Crop generated from damage segmentation.

Architecture

MobileNetV4

Classes

```
Minor

Moderate

Major
```

---

## Model 8 — Repair Cost Prediction

Input Features

Damage Type

Vehicle Part

Severity

Damage Area %

Vehicle Type

Vehicle Manufacturer

Vehicle Age

Model

Repair Method

Model

LightGBM

or

CatBoost

Output

Estimated Repair Cost

---

# Dataset Strategy

Instead of one dataset,

Create multiple Roboflow projects.

```
Vehicle AI Workspace

│

├── vehicle_detection

├── vehicle_segmentation

├── vehicle_part_segmentation

├── damage_segmentation

├── image_quality

├── orientation

├── severity

├── hard_negative

└── repair_cost_dataset
```

---

# Dataset 1 — Vehicle Detection

Annotation

Bounding Box

Classes

```
vehicle
```

Target

20,000 images

---

# Dataset 2 — Vehicle Segmentation

Annotation

Polygon

Classes

```
vehicle
```

Target

10,000 images

---

# Dataset 3 — Vehicle Part Segmentation

Annotation

Polygon

Important Rule

Annotate

ALL visible parts.

Not just damaged parts.

Clean vehicles should also have all visible parts annotated.

---

# Dataset 4 — Damage Segmentation

Annotation

Polygon

Classes

```
dent

scratch

broken

missing

shatter
```

Important Rule

Annotate only damaged pixels.

Never annotate full panel.

---

# Dataset 5 — Image Quality

Classification

Classes

```
good

blurry

dark

reflection

cropped

multiple_vehicle

occluded
```

---

# Dataset 6 — Orientation

Classification

Classes

```
front

rear

left

right

front_left

front_right

rear_left

rear_right
```

---

# Dataset 7 — Severity

Classification

Classes

```
minor

moderate

major
```

---

# Dataset 8 — Hard Negative Dataset

Purpose

Reduce false positives.

No annotations required.

Collect

- Clean vehicles
- Windshield reflections
- Chrome reflections
- Water droplets
- Mud
- Dust
- Logos
- Door handles
- Shadows
- Garage lighting
- Wet vehicles
- Tree reflections
- Human reflections
- Sun glare

Target

10,000 images

---

# Dataset Metadata

Maintain metadata CSV.

Columns

```
image_id

vehicle_type

manufacturer

model

year

camera_angle

weather

lighting

damage_type

vehicle_part

severity

repair_type

damage_area

reflection

occlusion
```

---

# Data Annotation Guidelines

## Vehicle Detection

Bounding box around primary vehicle.

---

## Vehicle Segmentation

Polygon around complete vehicle.

Exclude background.

---

## Vehicle Part Segmentation

Annotate

Every visible part.

Never skip undamaged parts.

Never guess hidden parts.

Annotate visible portion only.

---

## Damage Segmentation

Annotate

Only damaged pixels.

Dent

Small polygon.

Scratch

Thin polygon.

Broken

Broken component only.

Missing

Missing region only.

Shatter

Only cracked glass.

---

# Training Strategy

## Phase 1

Train every model independently.

---

## Phase 2

Evaluate

Precision

Recall

mAP

Confusion Matrix

---

## Phase 3

Run inference on

5,000+ unseen vehicle images.

Collect

False Positives

Examples

- Windshield reflections
- Chrome
- Logos
- Stickers
- Mud
- Water droplets
- Shadows
- Door handles

Add them to Hard Negative dataset.

Retrain.

Repeat until false positives become minimal.

This process is called Hard Negative Mining.

---

# Data Augmentation

Use Albumentations.

Recommended

- Horizontal Flip
- Random Brightness
- Contrast
- CLAHE
- Motion Blur
- Gaussian Noise
- Random Shadow
- Random Fog
- Rain
- JPEG Compression
- Color Jitter
- Perspective Transform
- Random Crop
- Random Scale
- Reflection Simulation

---

# Rule-Based Validation

Apply after damage detection.

Allowed Damage Mapping

| Vehicle Part | Allowed Damage |
|--------------|----------------|
| Windshield | Shatter |
| Rear Glass | Shatter |
| Side Window | Shatter |
| Door | Dent, Scratch |
| Hood | Dent, Scratch |
| Roof | Dent, Scratch |
| Fender | Dent, Scratch |
| Mirror | Broken, Missing |
| Headlight | Broken, Missing |
| Tail Light | Broken, Missing |
| Wheel | Missing |
| Tire | Missing |

Example

Reject

```
Dent on Windshield
```

Accept

```
Shatter on Windshield
```

---

# Damage-Part Mapping

Compute IoU

Between

Damage Mask

and

Vehicle Part Mask

Output

```
Dent

↓

Left Front Door

↓

Moderate

↓

Area 12%

↓

Estimated Cost ₹6,500
```

---

# Evaluation Metrics

Vehicle Detection

- mAP

Vehicle Segmentation

- IoU
- Dice Score

Vehicle Parts

- mIoU

Damage

- mAP
- mIoU

Severity

- Accuracy
- F1 Score

Cost

- MAE
- RMSE

Overall

- End-to-End Precision
- End-to-End Recall
- False Positive Rate
- Average Inference Time

---

# Deployment

Inference Pipeline

```
Image

↓

Quality Check

↓

Vehicle Detection

↓

Vehicle Segmentation

↓

Vehicle Part Segmentation

↓

Damage Segmentation

↓

Rule Validation

↓

Severity Classification

↓

Damage Area Calculation

↓

Repair Cost Prediction

↓

Insurance Report JSON
```

Deployment Targets

- AWS ECS
- EC2
- ONNX Runtime
- TensorRT (GPU)
- OpenVINO (CPU)

---

# Future Improvements

- Multi-view damage fusion
- Video-based inspection
- 3D damage estimation
- VIN recognition
- OCR for license plates
- LLM-generated claim summaries
- Human-in-the-loop verification
- Continual learning with approved claims
- Active learning for difficult edge cases

---

# Project Principles

1. One model = One task.
2. Never combine vehicle parts and damage into a single segmentation dataset.
3. Annotate all visible vehicle parts.
4. Annotate only damaged pixels.
5. Build a large hard-negative dataset.
6. Use rule-based validation to eliminate impossible predictions.
7. Continuously improve using hard negative mining and production feedback.
8. Optimize for explainability, maintainability, and lightweight deployment suitable for insurance workflows.