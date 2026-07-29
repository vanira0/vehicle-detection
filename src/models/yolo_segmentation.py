"""
YOLO11 Segmentation Model Wrapper.

This model bypasses the standard PyTorch training loop provided by Trainer
and directly uses the highly-optimized Ultralytics engine.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base import BaseDetector
from utils.config import Config
from .registry import register_model


# def _resolve_weights_path(weights: Optional[str], project_root: Optional[str] = None) -> str:
#     """Resolve a local checkpoint path or fall back to an available local YOLO11 weight."""
#     if not weights:
#         return "yolo11n-seg.pt"

#     if not project_root:
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#     if os.path.isabs(weights) or os.path.exists(weights):
#         return weights

#     project_path = os.path.join(project_root, weights)
#     if os.path.exists(project_path):
#         return project_path

#     if weights.startswith("yolo26"):
#         fallback_candidates = [
#             "yolo11s-seg.pt",
#             "yolo11m-seg.pt",
#             "yolo11n-seg.pt",
#             "yolo26n-seg.pt"
#         ]
#         for fallback in fallback_candidates:
#             fallback_path = os.path.join(project_root, fallback)
#             if os.path.exists(fallback_path):
#                 return fallback_path

#     return weights


def _apply_clahe_to_batch(trainer) -> None:
    """Apply CLAHE to each image in the Ultralytics training batch if available.

    Some Ultralytics versions expose the batch through ``trainer.batch`` while
    others use a slightly different object layout. This callback must stay
    defensive so training can continue even when the batch payload is missing.
    """
    batch = getattr(trainer, "batch", None)
    if batch is None:
        return

    imgs = None
    if isinstance(batch, dict):
        imgs = batch.get("img")
    elif hasattr(batch, "get"):
        imgs = batch.get("img")
    elif hasattr(batch, "img"):
        imgs = batch.img

    if imgs is None:
        return

    device = imgs.device
    orig_dtype = imgs.dtype

    # To uint8 numpy [B, H, W, C]
    imgs_np = (
        imgs.permute(0, 2, 3, 1).cpu().float().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(len(imgs_np)):
        lab = cv2.cvtColor(imgs_np[i], cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        imgs_np[i] = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Back to original dtype/device tensor [B, C, H, W]
    if isinstance(batch, dict):
        batch["img"] = (
            torch.from_numpy(imgs_np).float() / 255.0
        ).permute(0, 3, 1, 2).to(device=device, dtype=orig_dtype)
    elif hasattr(batch, "__setitem__"):
        batch["img"] = (
            torch.from_numpy(imgs_np).float() / 255.0
        ).permute(0, 3, 1, 2).to(device=device, dtype=orig_dtype)
    elif hasattr(batch, "img"):
        batch.img = (
            torch.from_numpy(imgs_np).float() / 255.0
        ).permute(0, 3, 1, 2).to(device=device, dtype=orig_dtype)

@register_model("yolo11_seg")
@register_model("yolo26_seg")
class YOLO11SegmentationWrapper(BaseDetector):
    """
    Wrapper for YOLO11 Segmentation models.
    Provides standard inference methods, but training should be invoked natively.
    """
    
    def __init__(self):
        if YOLO is None:
            raise ImportError(
                "Ultralytics package is missing. "
                "Please install it with: pip install ultralytics"
            )
        self._model = None
        
    def build(self, model_config: Config) -> nn.Module:
        """
        Construct the model.
        YOLO uses its own wrapper, so we return self, or the YOLO model directly.
        Here we initialize the YOLO model from a weight file (e.g. 'yolo11n-seg.pt').
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        weights = getattr(model_config, "weights", "yolo11n-seg.pt")
        # weights = _resolve_weights_path(weights, project_root)
        self._model = YOLO(weights)
        
        # We return a dummy nn.Module to satisfy the BaseDetector type signature
        # if the Trainer strictly expects nn.Module, OR we can return the wrapper.
        # Since we bypass the trainer for YOLO, this is fine.
        return self._model.model

    def train_native(self, config: Config):
        """
        Use Ultralytics' native training loop instead of the custom Trainer.
        """
        data_yaml = getattr(config.data, "yaml_path", "data.yaml")
        epochs = getattr(config.training, "epochs", 50)
        batch_size = getattr(config.training, "batch_size", 4)
        img_size = getattr(config.data, "image_size", 640)
        
        # We can extract optimizer params as well if needed
        lr = getattr(config.training.optimizer, "lr", 0.01)
        
        # Resolve project root dynamically (3 levels up from src/models/yolo_segmentation.py)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Ensure the project output path is absolute
        runs_dir = getattr(config.output, "runs_dir", "runs")
        if not os.path.isabs(runs_dir):
            runs_dir = os.path.join(project_root, runs_dir)
            
        data_yaml = getattr(config.data, "yaml_path", "data.yaml")
        if not os.path.isabs(data_yaml):
            data_yaml = os.path.join(project_root, data_yaml)

        # Extract any extra YOLO specific kwargs
        yolo_kwargs = {}
        if hasattr(config.training, "yolo_kwargs"):
            yolo_kwargs = config.training.yolo_kwargs.to_dict()
            
        # Build training arguments
        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "lr0": lr,
            "project": runs_dir,
            "name": getattr(config, "experiment_name", "yolo_experiment"),
            "task": "segment",
            "exist_ok": True
        }
        
        # Merge any custom kwargs provided by the user
        train_args.update(yolo_kwargs)

        self._model.add_callback(
            "on_train_batch_start", _apply_clahe_to_batch
        )

        # Launch YOLO training
        results = self._model.train(**train_args)
        return results

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Not implemented because training is handled natively by train_native().
        """
        raise NotImplementedError("YOLO11 uses native training. Call train_native().")

    def post_process(
        self,
        predictions: Any,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Converts Ultralytics Results objects into standard format.
        """
        results_list = []
        for result in predictions:
            # result is an ultralytics.engine.results.Results object
            processed = {
                "boxes": np.zeros((0, 4)),
                "labels": np.zeros((0,), dtype=int),
                "scores": np.zeros((0,)),
                "masks": np.zeros((0, result.orig_shape[0], result.orig_shape[1]))
            }
            
            if result.boxes:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter by confidence
                mask_conf = scores >= confidence_threshold
                boxes = boxes[mask_conf]
                scores = scores[mask_conf]
                labels = labels[mask_conf]
                
                processed["boxes"] = boxes
                processed["labels"] = labels
                processed["scores"] = scores
                
                if result.masks:
                    # masks.data contains the tensor masks, masks.xy contains polygon coordinates
                    masks = result.masks.data.cpu().numpy()
                    masks = masks[mask_conf]
                    
                    # Ensure masks match original image size
                    if masks.shape[1:] != result.orig_shape:
                        # Resize masks if necessary (using OpenCV)
                        import cv2
                        resized_masks = []
                        for m in masks:
                            m_resized = cv2.resize(m, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                            resized_masks.append(m_resized)
                        masks = np.array(resized_masks)
                        
                    processed["masks"] = masks
            
            results_list.append(processed)
            
        return results_list

    def predict(self, images, conf=0.5):
        """
        Run inference using YOLO11 natively.
        images can be a list of paths, numpy arrays, or PIL images.
        """
        if self._model is None:
            raise RuntimeError("Model is not built. Call build() first.")
            
        results = self._model.predict(images, conf=conf)
        return self.post_process(results, confidence_threshold=conf)
