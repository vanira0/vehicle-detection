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

    def build(self, model_config: Config) -> nn.Module:
        variant = getattr(model_config, "backbone", "yolo11n-cls")
        pretrained = getattr(model_config, "pretrained", True)
        weights = f"{variant}.pt" if pretrained else f"{variant}.yaml"
        self._model = YOLO(weights)
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

        results = self._model.train(**train_args)
        return results

    def compute_loss(self, model, images, labels):
        raise NotImplementedError("YOLO11 uses native training. Call train_native().")

    def predict(self, model, image):
        results = self._model.predict(image, verbose=False)
        result = results[0]
        predicted_class = result.probs.top1
        confidence = result.probs.top1conf.item()
        probs = result.probs.data.cpu().tolist()

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
