"""
Evaluator for running model evaluation on test datasets.

Supports both classification and detection evaluation using
COCO-style metrics via pycocotools when available.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
)


class Evaluator:
    """
    Run evaluation on a trained model checkpoint.

    Handles both classification (Gatekeeper) and detection
    (Damage/Parts) models.

    Usage:
        evaluator = Evaluator(model_type="detection")
        metrics = evaluator.evaluate(model, test_loader, device)
    """

    def __init__(
        self,
        model_type: str = "detection",
        num_classes: int = 2,
        iou_thresholds: Optional[List[float]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model_type: "classification" or "detection".
            num_classes: Number of classes.
            iou_thresholds: IoU thresholds for mAP calculation.
            confidence_threshold: Min confidence for detection predictions.
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ]
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given data loader.

        Args:
            model: The model to evaluate (must be in eval mode or will be set).
            data_loader: DataLoader for the evaluation dataset.
            device: Device to run evaluation on.

        Returns:
            Dict of metric names to values.
        """
        model.eval()

        if self.model_type == "classification":
            return self._evaluate_classification(model, data_loader, device)
        else:
            return self._evaluate_detection(model, data_loader, device)

    def _evaluate_classification(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate a classification model."""
        all_preds = []
        all_labels = []

        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return compute_classification_metrics(
            all_preds, all_labels, self.num_classes
        )

    def _evaluate_detection(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate a detection/segmentation model."""
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            predictions = model(images)

            for pred in predictions:
                scores = pred["scores"].cpu().numpy()
                keep = scores >= self.confidence_threshold
                all_predictions.append({
                    "boxes": pred["boxes"][keep].cpu().numpy(),
                    "labels": pred["labels"][keep].cpu().numpy(),
                    "scores": scores[keep],
                })

            for target in targets:
                all_targets.append({
                    "boxes": target["boxes"].cpu().numpy(),
                    "labels": target["labels"].cpu().numpy(),
                })

        return compute_detection_metrics(
            all_predictions,
            all_targets,
            iou_thresholds=self.iou_thresholds,
            num_classes=self.num_classes,
        )

    def evaluate_and_save(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        output_path: str,
    ) -> Dict[str, float]:
        """Evaluate and save results to JSON."""
        metrics = self.evaluate(model, data_loader, device)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics
