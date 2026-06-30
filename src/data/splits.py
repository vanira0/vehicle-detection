"""
Dataset splitting utilities.

Provides reproducible train/val/test splits for both classification
(folder-based) and COCO-format datasets. Split indices are saved
to JSON for reproducibility.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def create_splits(
    image_ids: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    labels: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, List]:
    """
    Split image IDs into train/val/test sets.

    Supports stratified splitting when labels are provided,
    ensuring each split has proportional class representation.

    Args:
        image_ids: List of image IDs (ints or strings).
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        labels: Optional labels for stratified splitting.
        save_path: Optional path to save split indices as JSON.

    Returns:
        Dict with keys "train", "val", "test" mapping to lists of image IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = random.Random(seed)

    if labels is not None:
        splits = _stratified_split(
            image_ids, labels, train_ratio, val_ratio, test_ratio, rng
        )
    else:
        splits = _random_split(
            image_ids, train_ratio, val_ratio, test_ratio, rng
        )

    # Save split indices for reproducibility
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert to serializable format
        serializable = {
            k: [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in v]
            for k, v in splits.items()
        }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)

    return splits


def _random_split(
    ids: List,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List]:
    """Simple random split."""
    ids = list(ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def _stratified_split(
    ids: List,
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List]:
    """Stratified split preserving class proportions in each split."""
    # Group IDs by label
    groups = defaultdict(list)
    for id_, label in zip(ids, labels):
        groups[label].append(id_)

    train_ids, val_ids, test_ids = [], [], []

    for label in sorted(groups.keys()):
        group = groups[label]
        rng.shuffle(group)

        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_ids.extend(group[:n_train])
        val_ids.extend(group[n_train : n_train + n_val])
        test_ids.extend(group[n_train + n_val :])

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def load_splits(path: str) -> Dict[str, List]:
    """
    Load previously saved split indices from JSON.

    Args:
        path: Path to the splits JSON file.

    Returns:
        Dict with keys "train", "val", "test".
    """
    with open(path, "r") as f:
        return json.load(f)


def split_coco_annotations(
    annotation_file: str,
    splits: Dict[str, List],
    output_dir: str,
) -> Dict[str, str]:
    """
    Split a COCO annotation file into separate train/val/test annotation files
    based on image ID splits.

    Args:
        annotation_file: Path to the full COCO JSON annotation file.
        splits: Dict from create_splits() with image ID lists.
        output_dir: Directory to save split annotation files.

    Returns:
        Dict mapping split names to output file paths.
    """
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}

    for split_name, split_ids in splits.items():
        split_ids_set = set(split_ids)

        # Filter images
        split_images = [
            img for img in coco_data["images"]
            if img["id"] in split_ids_set
        ]

        # Filter annotations
        split_annotations = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in split_ids_set
        ]

        # Create split COCO JSON
        split_data = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data["categories"],
        }

        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w") as f:
            json.dump(split_data, f)

        output_paths[split_name] = output_path

    return output_paths
