"""
Dataset implementations for the vehicle damage detection pipeline.

Two dataset types:
    1. ClassificationDataset  — Folder-based for the Gatekeeper (Stage 1)
    2. COCOSegmentationDataset — COCO JSON for Damage & Parts segmentation (Stages 2 & 3)

Both return standardized outputs compatible with PyTorch DataLoader.
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    Folder-based image classification dataset for the Gatekeeper model.

    Expected directory structure:
        root/
            damaged/
                img001.jpg
                img002.jpg
            undamaged/
                img003.jpg
                img004.jpg

    Args:
        root: Path to the dataset root directory.
        transform: Optional Albumentations transform pipeline.
        class_names: Optional explicit list of class names. If None,
                     sorted subdirectory names are used.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = root
        self.transform = transform

        # Discover classes from subdirectories
        if class_names is None:
            self.class_names = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Build list of (image_path, label) tuples
        self.samples: List[Tuple[str, int]] = []
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class COCOSegmentationDataset(Dataset):
    """
    COCO-format instance segmentation dataset for Damage and Part models.

    Args:
        root: Path to the directory containing images.
        annotation_file: Path to the COCO JSON annotation file.
        transform: Optional Albumentations transform pipeline.
            Must be configured with BboxParams and mask support.
        min_area: Minimum mask area (in pixels) to include an annotation.
            Filters out very small/degenerate annotations.
    """

    def __init__(
        self,
        root: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
        min_area: int = 10,
    ):
        self.root = root
        self.transform = transform
        self.min_area = min_area

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        # Filter out images with no annotations
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        # Build category mapping
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_contiguous = {
            cat["id"]: i + 1 for i, cat in enumerate(cats)
        }  # 0 is reserved for background
        self.class_names = ["background"] + [cat["name"] for cat in cats]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract boxes, masks, labels, areas
        boxes = []
        masks = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Filter out annotations with missing or empty segmentations
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            
            if isinstance(ann["segmentation"], list):
                # A valid polygon in COCO needs at least 6 coordinates (3 points)
                valid_segm = [poly for poly in ann["segmentation"] if len(poly) >= 6]
                if not valid_segm:
                    continue
                ann["segmentation"] = valid_segm

            # Get binary mask from annotation
            mask = self.coco.annToMask(ann)
            area = mask.sum()

            if area < self.min_area:
                continue

            # Bounding box: COCO format [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            masks.append(mask)
            labels.append(self.cat_id_to_contiguous[ann["category_id"]])
            areas.append(area)
            iscrowd.append(ann.get("iscrowd", 0))

        # Handle images with no valid annotations after filtering
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            labels = np.array([], dtype=np.int64)
            areas = np.array([], dtype=np.float32)
            iscrowd = np.array([], dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            masks = np.array(masks, dtype=np.uint8)
            labels = np.array(labels, dtype=np.int64)
            areas = np.array(areas, dtype=np.float32)
            iscrowd = np.array(iscrowd, dtype=np.int64)

        # Apply augmentations
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=image,
                masks=[masks[i] for i in range(len(masks))],
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
            image = transformed["image"]
            if transformed["bboxes"]:
                boxes = np.array(transformed["bboxes"], dtype=np.float32)
                masks = np.array(transformed["masks"], dtype=np.uint8)
                labels = np.array(transformed["labels"], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                masks = np.zeros(
                    (0, image.shape[-2] if isinstance(image, torch.Tensor) else image.shape[0],
                     image.shape[-1] if isinstance(image, torch.Tensor) else image.shape[1]),
                    dtype=np.uint8
                )
                labels = np.array([], dtype=np.int64)
        elif self.transform:
            transformed = self.transform(
                image=image,
                masks=[],
                bboxes=[],
                labels=[],
            )
            image = transformed["image"]

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    """
    Custom collate function for detection datasets.
    Mask R-CNN expects a list of images and a list of target dicts
    (not stacked tensors), because images/targets can have different sizes.
    """
    return tuple(zip(*batch))


# def download_roboflow_dataset(
#     api_key: str,
#     workspace: str,
#     project: str,
#     version: int,
#     export_format: str = "yolov8",
#     dest_dir: Optional[str] = None
# ) -> str:
#     """
#     Downloads a dataset from Roboflow.
    
#     Args:
#         api_key: Your Roboflow API Key.
#         workspace: The Roboflow workspace name.
#         project: The Roboflow project name.
#         version: The dataset version number.
#         export_format: The format to export to (e.g., 'yolov8', 'coco').
#         dest_dir: Destination directory for the dataset.
        
#     Returns:
#         The path to the downloaded dataset.
#     """
#     try:
#         from roboflow import Roboflow
#     except ImportError:
#         raise ImportError("Please install the roboflow package using `pip install roboflow`")
        
#     rf = Roboflow(api_key=api_key)
#     proj = rf.workspace(workspace).project(project)
#     vers = proj.version(version)
    
#     # download to the current working directory or dest_dir
#     if dest_dir:
#         dataset = vers.download(export_format, location=dest_dir)
#     else:
#         dataset = vers.download(export_format)
        
#     return dataset.location

