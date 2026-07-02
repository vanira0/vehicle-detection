"""
Evaluation metrics for all pipeline stages.

Supports:
    - Classification metrics (accuracy, precision, recall, F1) for Gatekeeper
    - Detection metrics (mAP@0.5, mAP@0.5:0.95) for Damage & Parts models
    - Mask IoU metrics for segmentation quality
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute classification metrics for the Gatekeeper model.

    Args:
        predictions: Predicted class indices (N,).
        labels: Ground truth class indices (N,).
        num_classes: Number of classes.

    Returns:
        Dict with accuracy, precision, recall, f1 (macro-averaged).
    """
    # Accuracy
    accuracy = (predictions == labels).mean()

    # Per-class precision, recall, F1
    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        tp = ((predictions == cls) & (labels == cls)).sum()
        fp = ((predictions == cls) & (labels != cls)).sum()
        fn = ((predictions != cls) & (labels == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


def compute_detection_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_thresholds: Optional[List[float]] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute detection metrics (mAP) for segmentation models.

    Uses a simplified mAP calculation. For full COCO-style mAP,
    use the COCOEvaluator in evaluator.py which wraps pycocotools.

    Args:
        predictions: List of prediction dicts per image with
                     boxes, labels, scores.
        targets: List of target dicts per image with boxes, labels.
        iou_thresholds: IoU thresholds for AP calculation.
        num_classes: Number of object classes (excluding background).

    Returns:
        Dict with mAP at various thresholds and per-class AP.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    all_aps = {t: [] for t in iou_thresholds}

    for cls in range(1, num_classes):  # Skip background (0)
        for threshold in iou_thresholds:
            ap = _compute_ap_for_class(predictions, targets, cls, threshold)
            all_aps[threshold].append(ap)

    # Compute mAP at various thresholds
    results = {}
    for threshold in iou_thresholds:
        if all_aps[threshold]:
            results[f"AP_{int(threshold * 100)}"] = float(np.mean(all_aps[threshold]))

    # Summary metrics
    if 0.5 in iou_thresholds:
        results["mAP_50"] = results.get("AP_50", 0.0)
    if 0.75 in iou_thresholds:
        results["mAP_75"] = results.get("AP_75", 0.0)

    # mAP@[0.5:0.95]
    all_thresholds_ap = [
        results.get(f"AP_{int(t * 100)}", 0.0) for t in iou_thresholds
    ]
    results["mAP_50_95"] = float(np.mean(all_thresholds_ap)) if all_thresholds_ap else 0.0

    return results


def _compute_ap_for_class(
    predictions: List[Dict],
    targets: List[Dict],
    class_id: int,
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at a given IoU threshold.

    Uses the 11-point interpolation method.
    """
    # Collect all predictions and ground truths for this class
    all_scores = []
    all_matches = []
    num_gt = 0

    for pred, gt in zip(predictions, targets):
        pred_boxes = pred.get("boxes", np.array([]))
        pred_labels = pred.get("labels", np.array([]))
        pred_scores = pred.get("scores", np.array([]))

        gt_boxes = gt.get("boxes", np.array([]))
        gt_labels = gt.get("labels", np.array([]))

        # Filter by class
        if len(pred_labels) > 0:
            pred_mask = pred_labels == class_id
            pred_boxes_cls = pred_boxes[pred_mask] if isinstance(pred_boxes, np.ndarray) else pred_boxes[pred_mask].numpy()
            pred_scores_cls = pred_scores[pred_mask] if isinstance(pred_scores, np.ndarray) else pred_scores[pred_mask].numpy()
        else:
            pred_boxes_cls = np.array([])
            pred_scores_cls = np.array([])

        if len(gt_labels) > 0:
            gt_mask = (gt_labels == class_id) if isinstance(gt_labels, np.ndarray) else (gt_labels.numpy() == class_id)
            gt_boxes_cls = gt_boxes[gt_mask] if isinstance(gt_boxes, np.ndarray) else gt_boxes[gt_mask].numpy()
        else:
            gt_boxes_cls = np.array([])

        num_gt += len(gt_boxes_cls)

        if len(pred_boxes_cls) == 0:
            continue

        # Sort predictions by score (descending)
        sorted_idx = np.argsort(-pred_scores_cls)
        pred_boxes_cls = pred_boxes_cls[sorted_idx] if len(pred_boxes_cls) > 0 else pred_boxes_cls
        pred_scores_cls = pred_scores_cls[sorted_idx]

        gt_matched = np.zeros(len(gt_boxes_cls), dtype=bool)

        for i in range(len(pred_boxes_cls)):
            all_scores.append(pred_scores_cls[i])

            if len(gt_boxes_cls) == 0:
                all_matches.append(False)
                continue

            # Compute IoU with all ground truth boxes
            ious = _compute_box_iou(
                pred_boxes_cls[i:i+1], gt_boxes_cls
            )[0]

            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_threshold and not gt_matched[best_iou_idx]:
                all_matches.append(True)
                gt_matched[best_iou_idx] = True
            else:
                all_matches.append(False)

    if num_gt == 0:
        return 0.0

    if len(all_matches) == 0:
        return 0.0

    # Sort by score
    sorted_idx = np.argsort(-np.array(all_scores))
    matches = np.array(all_matches, dtype=bool)[sorted_idx]

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(matches)
    fp_cumsum = np.cumsum(~matches)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # 11-point interpolation
    ap = 0.0
    for r_threshold in np.arange(0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= r_threshold]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= 11.0

    return float(ap)


def _compute_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: (N, 4) array of boxes in [x1, y1, x2, y2] format.
        boxes2: (M, 4) array of boxes.

    Returns:
        (N, M) array of IoU values.
    """
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

    return intersection / np.maximum(union, 1e-6)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        mask1: Binary mask (H, W).
        mask2: Binary mask (H, W).

    Returns:
        IoU value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / max(union, 1e-6))
