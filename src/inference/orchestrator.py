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
        angle_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Map each damage detection to the car part it overlaps with.

        Args:
            damage_predictions: Dict with keys: masks (N, H, W),
                                labels (N,), scores (N,).
            part_predictions: Dict with keys: masks (M, H, W),
                              labels (M,), scores (M,).
            angle_name: Optional string representing the view angle of the image.

        Returns:
            List of finding dicts, each containing:
                - damage, damage_type: str
                - corr part, corr_part, car_part, body_part, bodyPart: str
                - angle, angle_name: str
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
        part_boxes = part_predictions.get("boxes", np.array([]))

        num_damages = max(len(damage_masks), len(damage_boxes))
        num_parts = max(len(part_masks), len(part_boxes))
        if num_damages == 0 or num_parts == 0:
            return findings

        # Normalise all masks to a common spatial resolution if masks are present
        normalised_part_masks = []
        if len(damage_masks) > 0 and len(part_masks) > 0:
            target_h, target_w = damage_masks[0].shape[:2]
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

        for i in range(num_damages):
            d_label = int(damage_labels[i]) if i < len(damage_labels) else 0
            d_score = float(damage_scores[i]) if i < len(damage_scores) else 1.0
            damage_type_str = self._get_class_name(d_label, self.damage_classes)
            d_box = damage_boxes[i] if i < len(damage_boxes) else None

            use_masks = (len(damage_masks) > i and len(normalised_part_masks) > 0)
            if use_masks:
                d_mask = damage_masks[i].astype(bool)
                d_area = d_mask.sum()
            elif d_box is not None:
                d_area = max((d_box[2] - d_box[0]) * (d_box[3] - d_box[1]), 1e-6)
            else:
                d_area = 1.0

            matched_any = False

            for j in range(num_parts):
                p_label = int(part_labels[j]) if j < len(part_labels) else 0
                p_score = float(part_scores[j]) if j < len(part_scores) else 1.0

                if use_masks and j < len(normalised_part_masks):
                    p_mask = normalised_part_masks[j].astype(bool)
                    overlap = self._compute_mask_overlap(p_mask, d_mask)
                    p_area = p_mask.sum()
                elif d_box is not None and j < len(part_boxes):
                    p_box = part_boxes[j]
                    overlap = self._compute_box_overlap(p_box, d_box)
                    p_area = max((p_box[2] - p_box[0]) * (p_box[3] - p_box[1]), 1e-6)
                else:
                    continue

                if overlap >= self.iou_threshold:
                    matched_any = True
                    area_ratio = d_area / max(p_area, 1)
                    severity = self._classify_severity(area_ratio)
                    part_name_str = self._get_class_name(p_label, self.part_classes)

                    match_entry = {
                        "damage_index": i,
                        "part_index": j,
                        "damage_type": damage_type_str,
                        "body_part": part_name_str,
                        "angle": angle_name or "",
                        "severity": severity,
                        "damage_confidence": round(d_score, 3),
                        "part_confidence": round(p_score, 3),
                        "overlap_score": round(overlap, 3),
                        "damage_area_px": int(d_area),
                        "area_ratio": round(float(area_ratio), 3),
                    }
                    if d_box is not None:
                        match_entry["damage_box"] = d_box.tolist() if hasattr(d_box, "tolist") else list(d_box)

                    match_entry["description"] = (
                        f"{severity.title()} {damage_type_str} "
                        f"detected on the {part_name_str.replace('_', ' ').title()}"
                    )
                    findings.append(match_entry)

            # If no part met the threshold, fallback to the best overlapping part > 0.05 or 'unspecified'
            if not matched_any:
                best_j = -1
                best_overlap = 0.0
                for j in range(num_parts):
                    if use_masks and j < len(normalised_part_masks):
                        overlap = self._compute_mask_overlap(normalised_part_masks[j].astype(bool), d_mask)
                    elif d_box is not None and j < len(part_boxes):
                        overlap = self._compute_box_overlap(part_boxes[j], d_box)
                    else:
                        overlap = 0.0

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_j = j

                if best_j >= 0 and best_overlap > 0.05:
                    p_label = int(part_labels[best_j]) if best_j < len(part_labels) else 0
                    p_score = float(part_scores[best_j]) if best_j < len(part_scores) else 1.0
                    if use_masks and best_j < len(normalised_part_masks):
                        p_area = normalised_part_masks[best_j].astype(bool).sum()
                    elif best_j < len(part_boxes):
                        p_box = part_boxes[best_j]
                        p_area = max((p_box[2] - p_box[0]) * (p_box[3] - p_box[1]), 1e-6)
                    else:
                        p_area = 1.0
                    area_ratio = d_area / max(p_area, 1)
                    severity = self._classify_severity(area_ratio)
                    part_name_str = self._get_class_name(p_label, self.part_classes)
                else:
                    best_j = -1
                    best_overlap = 0.0
                    area_ratio = 0.0
                    severity = "minor"
                    part_name_str = "unspecified"
                    p_score = 0.0

                match_entry = {
                    "damage_index": i,
                    "part_index": best_j,
                    "damage_type": damage_type_str,
                    "body_part": part_name_str,
                    "angle": angle_name or "",
                    "severity": severity,
                    "damage_confidence": round(d_score, 3),
                    "part_confidence": round(p_score, 3),
                    "overlap_score": round(best_overlap, 3),
                    "damage_area_px": int(d_area),
                    "area_ratio": round(float(area_ratio), 3),
                }
                if d_box is not None:
                    match_entry["damage_box"] = d_box.tolist() if hasattr(d_box, "tolist") else list(d_box)
                match_entry["description"] = (
                    f"{severity.title()} {damage_type_str} "
                    f"detected on the {part_name_str.replace('_', ' ').title()}"
                )
                findings.append(match_entry)

        # Sort by severity (most severe first)
        severity_order = {"critical": 0, "severe": 1, "moderate": 2, "minor": 3}
        findings.sort(key=lambda f: severity_order.get(f["severity"], 4))

        return findings

    @staticmethod
    def _compute_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute how much of mask2 is covered by mask1 (Intersection over mask2 area)."""
        intersection = np.logical_and(mask1, mask2).sum()
        area2 = mask2.sum()
        return float(intersection / max(area2, 1e-6))

    @staticmethod
    def _compute_box_overlap(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute how much of box2 is covered by box1 (Intersection area over box2 area)."""
        damage_area = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1e-6)
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        overlap_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return float(overlap_area / damage_area)

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
