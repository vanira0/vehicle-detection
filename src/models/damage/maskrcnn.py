"""
Mask R-CNN with ResNet50-FPN backbone for damage instance segmentation.

This is the baseline strategy for Stage 2 (damage localization).
Uses torchvision's pretrained Mask R-CNN and replaces the prediction
heads for the target number of damage classes.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models.base import BaseDetector
from models.registry import register_model
from utils.config import Config


@register_model("maskrcnn_resnet50")
class MaskRCNNResNet50Damage(BaseDetector):
    """
    Mask R-CNN with ResNet50-FPN for instance segmentation.

    This model is shared between damage and parts stages — the stage
    is determined by the dataset and config, not the model class.

    Config options (in model section):
        - num_classes: Number of classes including background
        - pretrained: Use COCO-pretrained weights
        - trainable_backbone_layers: Number of backbone layers to fine-tune (0-5)
        - min_size: Minimum image size for the backbone
        - max_size: Maximum image size for the backbone
    """

    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 2)
        pretrained = getattr(model_config, "pretrained", True)
        trainable_layers = getattr(model_config, "trainable_backbone_layers", 3)
        min_size = getattr(model_config, "min_size", 800)
        max_size = getattr(model_config, "max_size", 1333)

        # Load pretrained Mask R-CNN
        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_layers,
            min_size=min_size,
            max_size=max_size,
        )

        # Replace box predictor head
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, num_classes
        )

        # Replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Mask R-CNN returns a dict of losses in training mode:
        loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg
        """
        return model(images, targets)

    def post_process(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Filter predictions by confidence and convert to numpy arrays.
        """
        results = []
        for pred in predictions:
            scores = pred["scores"].cpu().numpy()
            keep = scores >= confidence_threshold

            boxes = pred["boxes"][keep].cpu().numpy()
            labels = pred["labels"][keep].cpu().numpy()
            scores = scores[keep]
            masks = (pred["masks"][keep, 0].cpu().numpy() > 0.5).astype(np.uint8)

            results.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "masks": masks,
            })
        return results
