"""
YOLO11 / YOLOv8 Detection Model Wrapper for Instrument Cluster Warning Lights.

Registers two model names via the project registry:
    yolo11_detect  -- for YOLOv11 detection weights (e.g. yolo11s.pt)
    yolov8_detect  -- for YOLOv8 detection weights  (e.g. yolov8s.pt)

Key differences from YOLO11SegmentationWrapper:
    - task=detect (bounding box output, no masks)
    - post_process() returns boxes/labels/scores/names only
    - validate()                  runs post-training YOLO validation
    - export_visual_predictions() saves colour-annotated preview images
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
from .base import BaseDetector
from utils.config import Config
from .registry import register_model
logger = logging.getLogger(__name__)
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    # --- Critical Warnings (Red) --- BGR format ---
    "Battery_Charging_System":    (0,   0,   210),   # pure red
    "Engine_Oil_Pressure":        (0,   20,  220),   # red-orange tint
    "Engine_Coolant_Temperature": (30,  0,   200),   # deep red
    "Brake_System_Alert":         (0,   0,   180),   # dark red
    "Airbag_SRS":                 (10,  10,  230),   # bright red
    "Seat_Belt_Reminder":         (0,   40,  215),   # warm red
    "Power_Steering_Warning":     (20,  0,   190),   # crimson
    # --- Caution Warnings (Yellow/Amber) ---
    "Check_Engine_MIL":           (0,   200, 255),   # amber / yellow-orange
    "ABS":                        (0,   215, 255),   # amber
    "TPMS":                       (0,   210, 245),   # golden amber
    "Traction_Control":           (0,   200, 230),   # warm amber
    "Low_Fuel_Level":             (0,   190, 245),   # deep amber
    "Lane_Departure_Warning":     (20,  210, 240),   # amber-yellow
    "Glow_Plug_Indicator":        (0,   180, 220),   # orange-amber
    # --- Informational Indicators (Blue / Green / White) ---
    "High_Beam_Indicator":        (220, 120, 0  ),   # blue
    "Turn_Signal":                (0,   200, 0  ),   # green
    "Fog_Lights":                 (0,   180, 30 ),   # green (front) / amber (rear)
    "Cruise_Control_Active":      (200, 200, 200),   # white / light grey
    "Auto_Hold_EPB":              (0,   210, 60 ),   # bright green
}
_DEFAULT_COLOUR = (200, 200, 200)
def _colour_for_class(class_name: str) -> Tuple[int, int, int]:
    return CLASS_COLOURS.get(class_name, _DEFAULT_COLOUR)
def _resolve_weights(weights: Optional[str], project_root: str) -> str:
    if not weights:
        return "yolo11s.pt"
    if os.path.isabs(weights) or os.path.exists(weights):
        return weights
    candidate = os.path.join(project_root, weights)
    if os.path.exists(candidate):
        return candidate
    logger.info("Weights file %s not found locally; Ultralytics will attempt download.", weights)
    return weights
@register_model("yolo11_detect")
@register_model("yolov8_detect")
class InstrumentClusterDetector(BaseDetector):
    def __init__(self) -> None:
        if YOLO is None:
            raise ImportError("The ultralytics package is required. Install it with: pip install ultralytics")
        self._model: Optional[YOLO] = None
        self._class_names: List[str] = []
        self._project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    def build(self, model_config: Config) -> nn.Module:
        weights = getattr(model_config, "weights", "yolo11s.pt")
        weights = _resolve_weights(weights, self._project_root)
        logger.info("Loading YOLO detection weights from: %s", weights)
        self._model = YOLO(weights)
        return self._model.model
    def compute_loss(self, model: nn.Module, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("InstrumentClusterDetector uses native training. Call train_native().")
    def post_process(self, predictions: Any, confidence_threshold: float = 0.25, nms_threshold: float = 0.45) -> List[Dict[str, Any]]:
        results_list: List[Dict[str, Any]] = []
        for result in predictions:
            processed: Dict[str, Any] = {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int32), "scores": np.zeros((0,), dtype=np.float32), "names": []}
            if result.boxes is not None and len(result.boxes):
                boxes  = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                labels = result.boxes.cls.cpu().numpy().astype(np.int32)
                keep = scores >= confidence_threshold
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                class_names = result.names
                names = [class_names.get(int(l), str(l)) for l in labels]
                processed["boxes"]  = boxes
                processed["labels"] = labels
                processed["scores"] = scores
                processed["names"]  = names
            results_list.append(processed)
        return results_list
    def train_native(self, config: Config) -> Any:
        if self._model is None:
            raise RuntimeError("Call build() before train_native().")
        data_yaml = getattr(config.data, "yaml_path", "data.yaml")
        if not os.path.isabs(data_yaml):
            data_yaml = os.path.join(self._project_root, data_yaml)
        runs_dir = getattr(config.output, "runs_dir", "runs")
        if not os.path.isabs(runs_dir):
            runs_dir = os.path.join(self._project_root, runs_dir)
        epochs     = getattr(config.training, "epochs",     100)
        batch_size = getattr(config.training, "batch_size", 16)
        img_size   = getattr(config.data,     "image_size", 640)
        patience   = getattr(config.training, "patience",   20)
        device_cfg = getattr(config.model,    "device",     None)
        task       = getattr(config.model,    "task",       "detect")
        device: Any = device_cfg if device_cfg else (0 if torch.cuda.is_available() else "cpu")
        train_args: Dict[str, Any] = {
            "data":     data_yaml, "epochs":   epochs, "imgsz":    img_size,
            "batch":    batch_size, "patience": patience, "device":   device,
            "task":     task, "project":  runs_dir,
            "name":     getattr(config, "experiment_name", "warning_lights"),
            "exist_ok": True, "val":      True,
        }
        if hasattr(config.training, "yolo_kwargs"):
            extra = config.training.yolo_kwargs.to_dict()
            train_args.update(extra)
            logger.debug("Applied yolo_kwargs: %s", extra)
        logger.info("Starting Ultralytics training with args:")
        for k, v in train_args.items():
            logger.info("  %-22s = %s", k, v)
        results = self._model.train(**train_args)
        if hasattr(self._model, "names"):
            self._class_names = [self._model.names[i] for i in range(len(self._model.names))]
        return results
    def validate(self, data_yaml: Optional[str] = None, img_size: int = 640, conf: float = 0.25, iou: float = 0.45, device: Any = None) -> Dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call build() and train_native() first.")
        val_kwargs: Dict[str, Any] = {"imgsz": img_size, "conf": conf, "iou": iou, "plots": True}
        if data_yaml:
            val_kwargs["data"] = data_yaml
        if device is not None:
            val_kwargs["device"] = device
        logger.info("Running validation with: %s", val_kwargs)
        metrics = self._model.val(**val_kwargs)
        results = {
            "mAP50":     float(getattr(metrics, "box", metrics).map50),
            "mAP50_95":  float(getattr(metrics, "box", metrics).map),
            "precision": float(getattr(metrics, "box", metrics).mp),
            "recall":    float(getattr(metrics, "box", metrics).mr),
        }
        logger.info("Validation -- mAP50: %.4f | mAP50-95: %.4f | Precision: %.4f | Recall: %.4f", results["mAP50"], results["mAP50_95"], results["precision"], results["recall"])
        return results
    def export_visual_predictions(self, image_dir: str, output_dir: str, conf: float = 0.25, iou: float = 0.45, max_images: int = 50, device: Any = None) -> List[str]:
        if self._model is None:
            raise RuntimeError("Call build() before export_visual_predictions().")
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)[:max_images]
        if not image_paths:
            logger.warning("No images found in %s", image_dir)
            return []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        predict_kwargs: Dict[str, Any] = {"conf": conf, "iou": iou, "save": False, "verbose": False}
        if device is not None:
            predict_kwargs["device"] = device
        saved_paths: List[str] = []
        for img_path in image_paths:
            raw = cv2.imread(str(img_path))
            if raw is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            results = self._model.predict(str(img_path), **predict_kwargs)
            processed = self.post_process(results, confidence_threshold=conf)
            annotated = raw.copy()
            for det in processed:
                for box, label_idx, score, name in zip(det["boxes"], det["labels"], det["scores"], det["names"]):
                    x1, y1, x2, y2 = map(int, box)
                    colour = _colour_for_class(name)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                    caption = f"{name} {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                    cv2.putText(annotated, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            out_path = os.path.join(output_dir, img_path.name)
            cv2.imwrite(out_path, annotated)
            saved_paths.append(out_path)
            logger.debug("Saved annotated image: %s", out_path)
        logger.info("Visual predictions exported: %d images -> %s", len(saved_paths), output_dir)
        return saved_paths
    def predict(self, images: Any, conf: float = 0.25, iou: float = 0.45, device: Any = None) -> List[Dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Call build() first.")
        kwargs: Dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = self._model.predict(images, **kwargs)
        return self.post_process(results, confidence_threshold=conf)
