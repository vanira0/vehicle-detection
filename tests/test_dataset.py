"""
Tests for dataset classes and data utilities.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.splits import create_splits, split_coco_annotations
from utils.config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_from_dict(self):
        data = {"a": 1, "b": {"c": 2, "d": 3}}
        config = Config.from_dict(data)
        assert config.a == 1
        assert config.b.c == 2
        assert config.b.d == 3

    def test_to_dict(self):
        data = {"a": 1, "b": {"c": 2}}
        config = Config.from_dict(data)
        result = config.to_dict()
        assert result == data

    def test_get_with_dot_notation(self):
        data = {"training": {"optimizer": {"lr": 0.005}}}
        config = Config.from_dict(data)
        assert config.get("training.optimizer.lr") == 0.005
        assert config.get("training.optimizer.missing", 42) == 42

    def test_apply_overrides(self):
        data = {"training": {"epochs": 10, "lr": 0.01}}
        overrides = ["training.epochs=50", "training.lr=0.001"]
        result = Config._apply_overrides(data, overrides)
        assert result["training"]["epochs"] == 50
        assert result["training"]["lr"] == 0.001

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = Config._deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_contains(self):
        config = Config.from_dict({"a": 1, "b": 2})
        assert "a" in config
        assert "c" not in config


class TestSplits:
    """Tests for data splitting utilities."""

    def test_random_split_ratios(self):
        ids = list(range(100))
        splits = create_splits(ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

        # No overlap between splits
        all_ids = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_ids) == 100

    def test_stratified_split(self):
        ids = list(range(100))
        labels = [0] * 50 + [1] * 50
        splits = create_splits(ids, labels=labels)

        # All IDs should be present
        all_ids = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_ids) == 100

    def test_save_and_load_splits(self):
        ids = list(range(50))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "splits.json")
            splits = create_splits(ids, save_path=save_path)
            assert os.path.exists(save_path)

            from data.splits import load_splits
            loaded = load_splits(save_path)
            assert len(loaded["train"]) == len(splits["train"])

    def test_reproducibility(self):
        ids = list(range(100))
        splits1 = create_splits(ids, seed=42)
        splits2 = create_splits(ids, seed=42)
        assert splits1["train"] == splits2["train"]

    def test_split_coco_annotations(self):
        """Test COCO annotation file splitting."""
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg"},
                {"id": 2, "file_name": "img2.jpg"},
                {"id": 3, "file_name": "img3.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [0, 0, 20, 20]},
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 15, 15]},
            ],
            "categories": [
                {"id": 1, "name": "dent"},
                {"id": 2, "name": "scratch"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ann_file = os.path.join(tmpdir, "annotations.json")
            with open(ann_file, "w") as f:
                json.dump(coco_data, f)

            splits = {"train": [1, 2], "val": [3], "test": []}
            output_dir = os.path.join(tmpdir, "splits")
            result = split_coco_annotations(ann_file, splits, output_dir)

            assert os.path.exists(result["train"])
            with open(result["train"]) as f:
                train_data = json.load(f)
            assert len(train_data["images"]) == 2
            assert len(train_data["annotations"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
