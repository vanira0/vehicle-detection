"""
Orchestrator — maps damage detections to car parts via IoU overlap.

Computes the intersection between damage masks and part masks to
produce human-readable findings like "Severe Dent on Left Front Fender".

Also classifies damage severity based on the damage mask area relative
to the part mask area.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class Orchestrator:
    """
    Maps damage detections to car parts using mask IoU overlap.

    Usage:
        orchestrator = Orchestrator(iou_threshold=0.3)
        findings = orchestrator.map_damage_to_parts(damage_preds, parts_preds)
    """

    # Severity thresholds based on damage area / part area ratio
    SEVERITY_THRESHOLDS = {
        "minor": 0.05,     # < 5% of part area
        "moderate": 0.15,  # 5–15% of part area
        "severe": 0.30,    # 15–30% of part area
        "critical": 1.0,   # > 30% of part area
    }

    def __init__(
        self,
        iou_threshold: float = 0.3,
        damage_classes: Optional[List[str]] = None,
        part_classes: Optional[List[str]] = None,
    ):
        self.iou_threshold = iou_threshold
        self.damage_classes = damage_classes or [
            "dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"
        ]
        self.part_classes = part_classes or [
            "Background", "Quarter-panel", "Front-wheel", "Back-window", "Trunk",
            "Front-door", "Rocker-panel", "Grille", "Windshield", "Front-window",
            "Back-door", "Headlight", "Back-wheel", "Back-windshield", "Hood", 
            "Fender", "Tail-light", "License-plate", "Front-bumper", "Back-bumper",
            "Mirror", "Roof"
        ]

    def map_damage_to_parts(
        self,
        damage_predictions: Dict[str, np.ndarray],
        part_predictions: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Map each damage detection to the car part it overlaps with.

        Args:
            damage_predictions: Dict with keys: masks (N, H, W),
                                labels (N,), scores (N,).
            part_predictions: Dict with keys: masks (M, H, W),
                              labels (M,), scores (M,).

        Returns:
            List of finding dicts, each containing:
                - damage_type: str
                - car_part: str
                - severity: str
                - damage_confidence: float
                - part_confidence: float
                - overlap_score: float (IoU)
                - damage_area_px: int
                - description: str (human-readable)
        """
        findings = []

        damage_masks = damage_predictions.get("masks", np.array([]))
        damage_labels = damage_predictions.get("labels", np.array([]))
        damage_scores = damage_predictions.get("scores", np.array([]))
        damage_boxes = damage_predictions.get("boxes", np.array([]))

        part_masks = part_predictions.get("masks", np.array([]))
        part_labels = part_predictions.get("labels", np.array([]))
        part_scores = part_predictions.get("scores", np.array([]))

        if len(damage_masks) == 0 or len(part_masks) == 0:
            return findings


        # Normalise all masks to a common spatial resolution so that IoU is
        # valid even when damage and parts models output different sizes
        # (e.g. MaskRCNN at 1024×1024 vs YOLO resized to orig_shape).
        target_h, target_w = damage_masks[0].shape[:2]
        normalised_part_masks = []
        for pm in part_masks:
            if pm.shape[:2] != (target_h, target_w):
                pm = np.array(
                    __import__("cv2").resize(
                        pm.astype(np.uint8), (target_w, target_h),
                        interpolation=__import__("cv2").INTER_NEAREST,
                    ),
                    dtype=pm.dtype,
                )
            normalised_part_masks.append(pm)


        for i in range(len(damage_masks)):
            d_mask = damage_masks[i].astype(bool)
            d_label = int(damage_labels[i])
            d_score = float(damage_scores[i])
            d_area = d_mask.sum()

            best_match = None
            best_iou = 0.0


            for j, p_mask_raw in enumerate(normalised_part_masks):
                p_mask = p_mask_raw.astype(bool)
                p_label = int(part_labels[j])
                p_score = float(part_scores[j])

                # Compute IoU overlap
                iou = self._compute_mask_iou(d_mask, p_mask)

                if iou > self.iou_threshold and iou > best_iou:
                    p_area = p_mask.sum()
                    area_ratio = d_area / max(p_area, 1)
                    severity = self._classify_severity(area_ratio)

                    best_iou = iou
                    best_match = {
                        "damage_index": i,
                        "part_index": j,
                        "damage_type": self._get_class_name(d_label, self.damage_classes),
                        "car_part": self._get_class_name(p_label, self.part_classes),
                        "severity": severity,
                        "damage_confidence": round(d_score, 3),
                        "part_confidence": round(p_score, 3),
                        "overlap_score": round(iou, 3),
                        "damage_area_px": int(d_area),
                        "area_ratio": round(float(area_ratio), 3),
                    }
                    if len(damage_boxes) > i:
                        box = damage_boxes[i]
                        best_match["damage_box"] = box.tolist() if hasattr(box, "tolist") else box

            if best_match:
                # Build human-readable description
                best_match["description"] = (
                    f"{best_match['severity'].title()} {best_match['damage_type']} "
                    f"detected on the {best_match['car_part'].replace('_', ' ').title()}"
                )
                findings.append(best_match)

        # Sort by severity (most severe first)
        severity_order = {"critical": 0, "severe": 1, "moderate": 2, "minor": 3}
        findings.sort(key=lambda f: severity_order.get(f["severity"], 4))

        return findings

    @staticmethod
    def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection / max(union, 1e-6))

    def _classify_severity(self, area_ratio: float) -> str:
        """
        Classify damage severity based on damage area / part area ratio.
        """
        if area_ratio < self.SEVERITY_THRESHOLDS["minor"]:
            return "minor"
        elif area_ratio < self.SEVERITY_THRESHOLDS["moderate"]:
            return "moderate"
        elif area_ratio < self.SEVERITY_THRESHOLDS["severe"]:
            return "severe"
        else:
            return "critical"

    @staticmethod
    def _get_class_name(label: int, class_names: List[str]) -> str:
        """Safely get class name by index."""
        if 0 <= label < len(class_names):
            return class_names[label]
        return f"class_{label}"
