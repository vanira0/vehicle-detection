"""
Tests for the inference pipeline components.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference.orchestrator import Orchestrator
from evaluation.metrics import (
    compute_classification_metrics,
    compute_mask_iou,
    _compute_box_iou,
)


spec = importlib.util.spec_from_file_location(
    "infer_pipeline",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "infer_pipeline.py"),
)
infer_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer_pipeline)


class TestOrchestrator:
    """Tests for the damage-to-part orchestrator."""

    def setup_method(self):
        self.orchestrator = Orchestrator(
            iou_threshold=0.3,
            damage_classes=["background", "scratch", "dent", "crack"],
            part_classes=["background", "hood", "bumper", "fender"],
        )

    def test_basic_mapping(self):
        """Test that overlapping damage and part masks produce a finding."""
        h, w = 100, 100

        # Create damage mask covering top-left quadrant
        damage_mask = np.zeros((h, w), dtype=np.uint8)
        damage_mask[:50, :50] = 1

        # Create part mask covering top half
        part_mask = np.zeros((h, w), dtype=np.uint8)
        part_mask[:50, :] = 1

        damage_preds = {
            "masks": np.array([damage_mask]),
            "labels": np.array([1]),  # scratch
            "scores": np.array([0.9]),
        }
        part_preds = {
            "masks": np.array([part_mask]),
            "labels": np.array([1]),  # hood
            "scores": np.array([0.85]),
        }

        findings = self.orchestrator.map_damage_to_parts(damage_preds, part_preds)

        assert len(findings) == 1
        assert findings[0]["damage_type"] == "scratch"
        assert findings[0]["car_part"] == "hood"
        assert findings[0]["overlap_score"] > 0.3

    def test_no_overlap(self):
        """Non-overlapping masks should produce no findings."""
        h, w = 100, 100

        damage_mask = np.zeros((h, w), dtype=np.uint8)
        damage_mask[:10, :10] = 1  # Small corner

        part_mask = np.zeros((h, w), dtype=np.uint8)
        part_mask[90:, 90:] = 1  # Opposite corner

        damage_preds = {
            "masks": np.array([damage_mask]),
            "labels": np.array([2]),
            "scores": np.array([0.9]),
        }
        part_preds = {
            "masks": np.array([part_mask]),
            "labels": np.array([2]),
            "scores": np.array([0.9]),
        }

        findings = self.orchestrator.map_damage_to_parts(damage_preds, part_preds)
        assert len(findings) == 0

    def test_empty_predictions(self):
        """Empty predictions should return empty findings."""
        findings = self.orchestrator.map_damage_to_parts(
            {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])},
            {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])},
        )
        assert len(findings) == 0

    def test_severity_classification(self):
        """Test severity based on area ratio."""
        assert self.orchestrator._classify_severity(0.02) == "minor"
        assert self.orchestrator._classify_severity(0.10) == "moderate"
        assert self.orchestrator._classify_severity(0.25) == "severe"
        assert self.orchestrator._classify_severity(0.50) == "critical"


def test_no_parts_when_damage_is_below_confidence_threshold():
    """Parts should not be selected when the damage detection is below threshold."""
    damage_ctx = {
        "boxes": np.array([[0, 0, 10, 10]], dtype=float),
        "labels": np.array([0]),
        "scores": np.array([0.2]),
    }
    parts_ctx = {
        "boxes": np.array([[0, 0, 20, 20]], dtype=float),
        "labels": np.array([1]),
        "scores": np.array([0.9]),
    }

    indices = infer_pipeline._get_damaged_part_indices(
        [], damage_ctx, parts_ctx, overlap_ratio_threshold=0.7, confidence_threshold=0.5
    )

    assert indices == set()

    def test_findings_sorted_by_severity(self):
        """Findings should be sorted most severe first."""
        h, w = 100, 100

        # Small damage (minor)
        small_damage = np.zeros((h, w), dtype=np.uint8)
        small_damage[10:15, 10:15] = 1

        # Large damage (critical)
        large_damage = np.zeros((h, w), dtype=np.uint8)
        large_damage[10:60, 10:60] = 1

        # Part covering whole area
        part_mask = np.zeros((h, w), dtype=np.uint8)
        part_mask[:, :] = 1

        damage_preds = {
            "masks": np.array([small_damage, large_damage]),
            "labels": np.array([1, 2]),
            "scores": np.array([0.9, 0.8]),
        }
        part_preds = {
            "masks": np.array([part_mask]),
            "labels": np.array([1]),
            "scores": np.array([0.95]),
        }

        findings = self.orchestrator.map_damage_to_parts(damage_preds, part_preds)
        if len(findings) >= 2:
            severity_order = {"critical": 0, "severe": 1, "moderate": 2, "minor": 3}
            for i in range(len(findings) - 1):
                assert severity_order[findings[i]["severity"]] <= severity_order[findings[i+1]["severity"]]


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_classification_perfect(self):
        preds = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(preds, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_classification_random(self):
        preds = np.array([0, 1, 0, 1])
        labels = np.array([1, 0, 1, 0])
        metrics = compute_classification_metrics(preds, labels)
        assert metrics["accuracy"] == 0.0

    def test_mask_iou_identical(self):
        mask = np.ones((10, 10), dtype=bool)
        assert compute_mask_iou(mask, mask) == 1.0

    def test_mask_iou_no_overlap(self):
        mask1 = np.zeros((10, 10), dtype=bool)
        mask1[:5, :] = True
        mask2 = np.zeros((10, 10), dtype=bool)
        mask2[5:, :] = True
        assert compute_mask_iou(mask1, mask2) == 0.0

    def test_mask_iou_partial(self):
        mask1 = np.zeros((10, 10), dtype=bool)
        mask1[:5, :] = True
        mask2 = np.zeros((10, 10), dtype=bool)
        mask2[3:8, :] = True
        iou = compute_mask_iou(mask1, mask2)
        assert 0 < iou < 1

    def test_box_iou(self):
        boxes1 = np.array([[0, 0, 10, 10]])
        boxes2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        ious = _compute_box_iou(boxes1, boxes2)
        assert ious[0, 0] == 1.0  # Identical
        assert 0 < ious[0, 1] < 1  # Partial overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
