"""
YOLOv8 instance segmentation model for damage detection.

Alternative strategy using Ultralytics YOLOv8's segmentation variant.
Significantly faster training and inference than Mask R-CNN, with
a different accuracy/speed tradeoff.

Requires: pip install ultralytics
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from models.base import BaseDetector
from models.registry import register_model
from utils.config import Config


@register_model("yolov8_seg")
class YOLOv8SegDamage(BaseDetector):
    """
    YOLOv8 instance segmentation wrapper.

    Wraps Ultralytics' YOLOv8 segmentation API to conform to the
    BaseDetector interface. This allows it to be used interchangeably
    with Mask R-CNN in the pipeline.

    Config options (in model section):
        - variant: YOLOv8 model size (yolov8n-seg, yolov8s-seg, etc.)
        - pretrained: Use COCO-pretrained weights
        - num_classes: Number of damage classes (no bg class needed)

    Note: YOLOv8 has its own training loop via model.train().
    The compute_loss method wraps this to match the BaseDetector interface,
    but for full YOLOv8 training features, use the dedicated YOLOv8 training
    strategy in src/training/strategies.py.
    """

    def __init__(self):
        self._yolo_model = None

    def build(self, model_config: Config) -> nn.Module:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLOv8 requires the ultralytics package. "
                "Install with: pip install ultralytics"
            )

        variant = getattr(model_config, "variant", "yolov8m-seg")
        pretrained = getattr(model_config, "pretrained", True)

        if pretrained:
            model = YOLO(f"{variant}.pt")
        else:
            model = YOLO(f"{variant}.yaml")

        self._yolo_model = model
        # Return the underlying PyTorch model for compatibility
        return model.model

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Note: YOLOv8 has its own internal training loop.
        This method provides a compatibility layer for the generic Trainer.
        For best results with YOLOv8, use the YOLOv8TrainingStrategy.
        """
        # Stack images into batch
        if isinstance(images, list):
            batch = torch.stack(images)
        else:
            batch = images

        # Forward pass in training mode
        model.train()
        output = model(batch)

        # YOLOv8's model returns a tuple (loss, loss_items) in training mode
        if isinstance(output, tuple):
            loss = output[0]
            return {"loss": loss}
        else:
            return {"loss": torch.tensor(0.0)}

    def post_process(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Post-process YOLOv8 predictions to standardized format.
        YOLOv8's predict() method returns Results objects.
        """
        results = []
        for pred in predictions:
            if hasattr(pred, "boxes") and pred.boxes is not None:
                boxes = pred.boxes.xyxy.cpu().numpy()
                scores = pred.boxes.conf.cpu().numpy()
                labels = pred.boxes.cls.cpu().numpy().astype(int)

                keep = scores >= confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Extract masks if available
                masks = np.array([])
                if hasattr(pred, "masks") and pred.masks is not None:
                    masks = pred.masks.data[keep].cpu().numpy()

                results.append({
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                    "masks": masks,
                })
            else:
                results.append({
                    "boxes": np.array([]),
                    "labels": np.array([]),
                    "scores": np.array([]),
                    "masks": np.array([]),
                })
        return results

    def train_with_ultralytics(
        self,
        data_yaml: str,
        model_config: Config,
        training_config: Config,
    ) -> Dict[str, Any]:
        """
        Train using Ultralytics' native training API.
        This bypasses the generic Trainer for full YOLOv8 feature support.

        Args:
            data_yaml: Path to YOLOv8-format data.yaml file.
            model_config: Model config section.
            training_config: Training config section.

        Returns:
            Training results dict.
        """
        if self._yolo_model is None:
            self.build(model_config)

        results = self._yolo_model.train(
            data=data_yaml,
            epochs=getattr(training_config, "epochs", 100),
            imgsz=getattr(model_config, "image_size", 640),
            batch=getattr(training_config, "batch_size", 8),
            lr0=getattr(training_config.optimizer, "lr", 0.001),
            weight_decay=getattr(training_config.optimizer, "weight_decay", 0.0005),
            amp=getattr(training_config, "amp", True),
            device="0" if torch.cuda.is_available() else "cpu",
        )
        return results
