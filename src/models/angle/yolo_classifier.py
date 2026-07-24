from typing import Any, Dict
import os
import torch
import torch.nn as nn
from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

@register_model("yolo11n_cls_angle_classifier")
class YOLO11ClassificationWrapper(BaseClassifier):
    def __init__(self):
        if YOLO is None:
            raise ImportError("Ultralytics package is missing. Please install it.")
        self._model = None
        self._yolo_model = None

    def build(self, model_config: Config) -> nn.Module:
        variant = getattr(model_config, "backbone", "yolo11n-cls")
        pretrained = getattr(model_config, "pretrained", True)
        weights = f"{variant}.pt" if pretrained else f"{variant}.yaml"
        self._model = YOLO(weights)
        self._yolo_model = self._model
        return self._model.model

    def train_native(self, config: Config):
        epochs = getattr(config.training, "epochs", 30)
        batch_size = getattr(config.training, "batch_size", 16)
        img_size = getattr(config.data, "image_size", 224)
        lr = getattr(config.training.optimizer, "lr", 0.001)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        runs_dir = getattr(config.output, "runs_dir", "runs")
        if not os.path.isabs(runs_dir):
            runs_dir = os.path.join(project_root, runs_dir)
            
        data_root = getattr(config.data, "root", "rf_ds_angle-2")
        if not os.path.isabs(data_root):
            data_root = os.path.join(project_root, data_root)

        train_args = {
            "data": data_root,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "lr0": lr,
            "project": runs_dir,
            "name": getattr(config, "experiment_name", "angle_yolo11n_cls_v1"),
            "task": "classify",
            "exist_ok": True
        }

        # Extract any extra YOLO specific kwargs
        yolo_kwargs = {}
        if hasattr(config.training, "yolo_kwargs"):
            yolo_kwargs = config.training.yolo_kwargs.to_dict()
            
        # Merge any custom kwargs provided by the user
        train_args.update(yolo_kwargs)

        yolo = self._model or self._yolo_model
        if yolo is None:
            raise RuntimeError("YOLO model instance not initialized for training.")
        results = yolo.train(**train_args)
        return results

    def compute_loss(self, model, images, labels):
        raise NotImplementedError("YOLO11 uses native training. Call train_native().")

    def predict(self, model, image):
        yolo_obj = self._model if self._model is not None else getattr(self, "_yolo_model", None)
        if yolo_obj is None and hasattr(model, "predict"):
            yolo_obj = model
        if yolo_obj is None:
            raise RuntimeError("No YOLO model instance found in wrapper for prediction.")

        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)

        results = yolo_obj.predict(image, verbose=False)
        result = results[0]

        if hasattr(result, "probs") and result.probs is not None:
            predicted_class = int(result.probs.top1)
            confidence = float(result.probs.top1conf.item())
            probs = result.probs.data.cpu().tolist() if hasattr(result.probs.data, "cpu") else []
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            import numpy as np
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            predicted_class = int(result.boxes.cls[best_idx].cpu().item())
            confidence = float(confidences[best_idx])
            probs = []
        else:
            predicted_class = 0
            confidence = 0.0
            probs = []

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": probs,
        }

    @staticmethod
    def freeze_backbone(model: nn.Module) -> None:
        pass

    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        pass
