"""
Tests for the model registry and model implementations.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.registry import MODEL_REGISTRY, get_model, register_model, list_models
from models.base import BaseDetector, BaseClassifier
from utils.config import Config


# Import model packages to trigger registration
import models.gatekeeper
import models.angle
import models.damage      # noqa: F401
import models.parts       # noqa: F401


class TestModelRegistry:
    """Tests for the model registration system."""

    def test_models_are_registered(self):
        """All expected models should be in the registry after imports."""
        expected_models = [
            "resnet50_classifier",
            "mobilenetv3_classifier",
            "maskrcnn_resnet50",
            "maskrcnn_swin",
            "yolov8_seg",
            "maskrcnn_resnet50_parts",
        ]
        for name in expected_models:
            assert name in MODEL_REGISTRY, f"Model '{name}' not registered"

    def test_get_model_returns_class(self):
        cls = get_model("resnet50_classifier")
        assert cls is not None

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="not found in registry"):
            get_model("nonexistent_model")

    def test_list_models(self):
        names = list_models()
        assert isinstance(names, list)
        assert len(names) >= 6

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_model("resnet50_classifier")
            class Duplicate:
                pass


class TestGatekeeperModels:
    """Tests for gatekeeper model build."""

    def test_resnet50_builds(self):
        config = Config.from_dict({
            "num_classes": 2,
            "pretrained": False,  # Don't download weights in tests
            "dropout": 0.5,
        })
        wrapper = get_model("resnet50_classifier")()
        model = wrapper.build(config)
        assert model is not None

        # Check output layer has correct num_classes
        import torch
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (1, 2)

    def test_mobilenet_builds(self):
        config = Config.from_dict({
            "num_classes": 2,
            "pretrained": False,
            "dropout": 0.2,
            "variant": "small",  # Use small variant for faster test
        })
        wrapper = get_model("mobilenetv3_classifier")()
        model = wrapper.build(config)
        assert model is not None

        import torch
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (1, 2)


class TestDetectionModels:
    """Tests for detection model build."""

    def test_maskrcnn_resnet50_builds(self):
        config = Config.from_dict({
            "num_classes": 7,
            "pretrained": False,
            "trainable_backbone_layers": 3,
            "min_size": 200,
            "max_size": 300,
        })
        wrapper = get_model("maskrcnn_resnet50")()
        model = wrapper.build(config)
        assert model is not None

    def test_maskrcnn_parts_builds(self):
        config = Config.from_dict({
            "num_classes": 16,
            "pretrained": False,
            "trainable_backbone_layers": 3,
            "min_size": 200,
            "max_size": 300,
        })
        wrapper = get_model("maskrcnn_resnet50_parts")()
        model = wrapper.build(config)
        assert model is not None

    def test_base_detector_interface(self):
        """Verify all detection models implement required methods."""
        detection_models = ["maskrcnn_resnet50", "maskrcnn_swin", "maskrcnn_resnet50_parts"]
        for name in detection_models:
            cls = get_model(name)
            wrapper = cls()
            assert hasattr(wrapper, "build")
            assert hasattr(wrapper, "compute_loss")
            assert hasattr(wrapper, "post_process")

    def test_base_classifier_interface(self):
        """Verify all classifier models implement required methods."""
        classifier_models = ["resnet50_classifier", "mobilenetv3_classifier"]
        for name in classifier_models:
            cls = get_model(name)
            wrapper = cls()
            assert hasattr(wrapper, "build")
            assert hasattr(wrapper, "compute_loss")
            assert hasattr(wrapper, "predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
