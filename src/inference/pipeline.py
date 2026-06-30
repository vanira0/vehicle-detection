"""
End-to-end 3-stage inference pipeline.

Orchestrates all three model stages:
    1. Gatekeeper — Is this a damaged car?
    2. Damage Segmentation — Where is the damage?
    3. Part Segmentation — Which parts are affected?
    4. Orchestrator — Map damage to parts via IoU overlap.

Produces a final structured report.
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from data.preprocessing import load_image, preprocess_for_detection, preprocess_image
from models.registry import get_model
from utils.config import Config
from utils.logger import setup_logger

from .orchestrator import Orchestrator


# Default class names (can be overridden via config)
DEFAULT_DAMAGE_CLASSES = [
    "background", "scratch", "dent", "crack",
    "broken_part", "shattered_glass", "tear",
]

DEFAULT_PART_CLASSES = [
    "background", "hood", "front_bumper", "rear_bumper",
    "left_fender", "right_fender", "left_front_door",
    "right_front_door", "left_rear_door", "right_rear_door",
    "trunk", "roof", "windshield", "rear_window",
    "headlight", "taillight",
]


class VehicleDamagePipeline:
    """
    End-to-end vehicle damage detection pipeline.

    Usage:
        pipeline = VehicleDamagePipeline(
            gatekeeper_checkpoint="runs/gatekeeper_v1/checkpoints/best.pth",
            damage_checkpoint="runs/damage_v1/checkpoints/best.pth",
            parts_checkpoint="runs/parts_v1/checkpoints/best.pth",
        )
        result = pipeline("test_car.jpg")
        print(result)
    """

    def __init__(
        self,
        gatekeeper_checkpoint: str,
        damage_checkpoint: str,
        parts_checkpoint: str,
        iou_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        damage_classes: Optional[List[str]] = None,
        part_classes: Optional[List[str]] = None,
    ):
        self.logger = setup_logger("pipeline")

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.confidence_threshold = confidence_threshold
        self.damage_classes = damage_classes or DEFAULT_DAMAGE_CLASSES
        self.part_classes = part_classes or DEFAULT_PART_CLASSES

        # Load models
        self.logger.info("Loading pipeline models...")
        self.gatekeeper, self.gatekeeper_wrapper = self._load_model(
            gatekeeper_checkpoint, "gatekeeper"
        )
        self.damage_model, self.damage_wrapper = self._load_model(
            damage_checkpoint, "damage"
        )
        self.parts_model, self.parts_wrapper = self._load_model(
            parts_checkpoint, "parts"
        )

        # Setup orchestrator
        self.orchestrator = Orchestrator(
            iou_threshold=iou_threshold,
            damage_classes=self.damage_classes,
            part_classes=self.part_classes,
        )

        self.logger.info(f"Pipeline ready on device: {self.device}")

    def _load_model(self, checkpoint_path: str, stage: str):
        """
        Load a model from a checkpoint file.

        The checkpoint contains the saved config, which tells us
        which model class to instantiate from the registry.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = Config.from_dict(checkpoint["config"])

        # Get model class from registry
        model_name = config.model.name
        model_wrapper = get_model(model_name)()

        # Build and load weights
        model = model_wrapper.build(config.model).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.logger.info(
            f"Loaded {stage} model: {model_name} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )
        return model, model_wrapper

    def __call__(self, image_path: str) -> Dict[str, Any]:
        """
        Run the full pipeline on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with:
                - status: "no_damage" | "damaged"
                - gatekeeper: Gatekeeper result dict
                - damage: Damage predictions (if damaged)
                - parts: Part predictions (if damaged)
                - findings: Orchestrator report (if damaged)
                - inference_time_ms: Total inference time
        """
        start_time = time.time()
        image_rgb = load_image(image_path)

        # === Stage 1: Gatekeeper ===
        gate_image = preprocess_image(image_rgb, target_size=224).to(self.device)
        gate_result = self.gatekeeper_wrapper.predict(self.gatekeeper, gate_image)

        result = {
            "image_path": image_path,
            "gatekeeper": gate_result,
        }

        if not gate_result["is_damaged"]:
            result["status"] = "no_damage"
            result["inference_time_ms"] = (time.time() - start_time) * 1000
            return result

        # === Stage 2 & 3: Damage + Part Segmentation ===
        det_image = preprocess_for_detection(image_rgb, target_size=1024).to(self.device)

        with torch.no_grad():
            damage_preds = self.damage_model([det_image])
            parts_preds = self.parts_model([det_image])

        # Post-process
        damage_results = self.damage_wrapper.post_process(
            damage_preds, self.confidence_threshold
        )[0]
        parts_results = self.parts_wrapper.post_process(
            parts_preds, self.confidence_threshold
        )[0]

        result["damage"] = {
            "num_detections": len(damage_results["labels"]),
            "labels": damage_results["labels"].tolist(),
            "scores": damage_results["scores"].tolist(),
        }
        result["parts"] = {
            "num_detections": len(parts_results["labels"]),
            "labels": parts_results["labels"].tolist(),
            "scores": parts_results["scores"].tolist(),
        }

        # === Stage 4: Orchestrate ===
        findings = self.orchestrator.map_damage_to_parts(
            damage_results, parts_results
        )

        result["status"] = "damaged"
        result["findings"] = findings
        result["inference_time_ms"] = (time.time() - start_time) * 1000

        self.logger.info(
            f"Pipeline result: {len(findings)} findings "
            f"({result['inference_time_ms']:.0f}ms)"
        )
        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Run the pipeline on multiple images."""
        return [self(path) for path in image_paths]
