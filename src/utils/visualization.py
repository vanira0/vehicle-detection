"""
Visualization utilities for masks, bounding boxes, and detection results.

Provides functions for overlaying segmentation masks, drawing bounding boxes,
and generating side-by-side comparison visualizations.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# Default color palette (BGR for OpenCV)
COLORS = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 255, 0),   # Spring Green
    (255, 128, 0),   # Azure
    (0, 128, 255),   # Orange
    (128, 0, 255),   # Violet
    (255, 0, 128),   # Rose
    (0, 255, 128),   # Turquoise
    (64, 224, 208),  # Teal
    (147, 20, 255),  # Deep Pink
    (250, 206, 135), # Light Sky Blue
    (180, 105, 255), # Hot Pink
]


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: Optional[List[str]] = None,
    scores: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """
    Overlay segmentation masks on an image with transparency.

    Args:
        image: Input image (H, W, 3) in BGR format.
        masks: Binary masks (N, H, W).
        labels: List of class names for each mask.
        scores: Confidence scores (N,).
        alpha: Transparency factor for mask overlay.
        score_threshold: Minimum score to display a mask.

    Returns:
        Image with overlaid masks.
    """
    result = image.copy()

    for i, mask in enumerate(masks):
        if scores is not None and scores[i] < score_threshold:
            continue

        color = COLORS[i % len(COLORS)]
        mask_bool = mask.astype(bool)

        # Overlay the mask with transparency
        result[mask_bool] = (
            alpha * np.array(color) + (1 - alpha) * result[mask_bool]
        ).astype(np.uint8)

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

        # Add label text
        if labels and contours:
            x, y, _, _ = cv2.boundingRect(contours[0])
            label_text = labels[i] if labels else f"Class {i}"
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            cv2.putText(
                result, label_text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    return result


def draw_bboxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[str]] = None,
    scores: Optional[np.ndarray] = None,
    score_threshold: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input image (H, W, 3) in BGR format.
        boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format.
        labels: List of class names for each box.
        scores: Confidence scores (N,).
        score_threshold: Minimum score to display a box.
        thickness: Line thickness.

    Returns:
        Image with drawn bounding boxes.
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_threshold:
            continue

        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        if labels:
            label_text = labels[i]
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            # Background rectangle for text
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                result, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1
            )
            cv2.putText(
                result, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

    return result


def visualize_pipeline_output(
    image: np.ndarray,
    damage_results: Dict,
    part_results: Dict,
    orchestrator_report: List[Dict],
    damage_classes: List[str],
    part_classes: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Create a multi-panel visualization showing the full pipeline output:
    original image, damage masks, part masks, and final mapped results.

    Args:
        image: Original input image (H, W, 3) BGR.
        damage_results: Predictions from damage model (masks, labels, scores).
        part_results: Predictions from parts model.
        orchestrator_report: Output from orchestrator mapping.
        damage_classes: List of damage class names.
        part_classes: List of part class names.
        save_path: Optional path to save the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Damage masks
    damage_vis = overlay_masks(
        image,
        damage_results.get("masks", np.array([])),
        [damage_classes[l] for l in damage_results.get("labels", [])],
        damage_results.get("scores"),
    )
    axes[0].imshow(cv2.cvtColor(damage_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Damage Segmentation")
    axes[0].axis("off")

    # Panel 2: Part masks
    part_vis = overlay_masks(
        image,
        part_results.get("masks", np.array([])),
        [part_classes[l] for l in part_results.get("labels", [])],
        part_results.get("scores"),
    )
    axes[1].imshow(cv2.cvtColor(part_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Part Segmentation")
    axes[1].axis("off")

    # Panel 3: Final mapped results text overlay
    combined = overlay_masks(
        image,
        damage_results.get("masks", np.array([])),
        [damage_classes[l] for l in damage_results.get("labels", [])],
        damage_results.get("scores"),
    )
    axes[2].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    report_text = "\n".join(
        f"{r.get('damage_type', '?')} on {r.get('car_part', '?')} "
        f"(IoU: {r.get('overlap_score', 0):.2f})"
        for r in orchestrator_report
    )
    axes[2].set_title("Mapped Results")
    axes[2].text(
        0.02, 0.98, report_text or "No damage mapped",
        transform=axes[2].transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
