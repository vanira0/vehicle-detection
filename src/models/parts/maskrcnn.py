"""
Mask R-CNN with ResNet50-FPN backbone for car part instance segmentation.

Stage 3: Segments car parts (hood, bumper, fender, doors, etc.).
Reuses the same Mask R-CNN architecture as the damage model but
registers under a separate name for clarity.
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


# Note: The "maskrcnn_resnet50" name is already registered by the damage module.
# For parts, use the same class OR register with a different name if you need
# parts-specific customizations. Since the architecture is identical and only
# the config/dataset differs, we re-export the damage version.
# If you need parts-specific behavior, uncomment and modify below.

@register_model("maskrcnn_resnet50_parts")
class MaskRCNNResNet50Parts(BaseDetector):
    """
    Mask R-CNN for car part segmentation.

    Identical architecture to the damage Mask R-CNN but registered
    separately to allow independent configuration.

    Default part classes (15 + background):
        hood, front_bumper, rear_bumper, left_fender, right_fender,
        left_front_door, right_front_door, left_rear_door, right_rear_door,
        trunk, roof, windshield, rear_window, headlight, taillight
    """

    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 16)  # 15 parts + bg
        pretrained = getattr(model_config, "pretrained", True)
        trainable_layers = getattr(model_config, "trainable_backbone_layers", 3)
        min_size = getattr(model_config, "min_size", 800)
        max_size = getattr(model_config, "max_size", 1333)

        weights = "DEFAULT" if pretrained else None
        
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        # Custom anchor boxes for car parts (long, thin rectangles)
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.2, 0.5, 1.0, 2.0, 5.0),) * 5
        )
        
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_layers,
            min_size=min_size,
            max_size=max_size,
            rpn_anchor_generator=anchor_generator,
            box_nms_thresh=0.4,
        )

        # Replace heads for part classes
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, num_classes
        )

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )

        return model

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = model(images, targets)
        
        # Rebalance loss weights to prioritize classification and bounding box regression
        loss_weights = {
            "loss_classifier": 2.0,
            "loss_box_reg": 2.0,
            "loss_mask": 0.5
        }
        
        weighted_loss_dict = {}
        for k, v in loss_dict.items():
            if k in loss_weights:
                weighted_loss_dict[k] = v * loss_weights[k]
            else:
                weighted_loss_dict[k] = v
                
        return weighted_loss_dict

    def post_process(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        results = []
        for pred in predictions:
            scores = pred["scores"].cpu().numpy()
            keep = scores >= confidence_threshold

            results.append({
                "boxes": pred["boxes"][keep].cpu().numpy(),
                "labels": pred["labels"][keep].cpu().numpy(),
                "scores": scores[keep],
                "masks": (pred["masks"][keep, 0].cpu().numpy() > 0.5).astype(np.uint8),
            })
        return results
