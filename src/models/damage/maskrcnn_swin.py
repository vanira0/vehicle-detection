"""
Mask R-CNN with Swin Transformer backbone for damage instance segmentation.

Alternative strategy that swaps the ResNet50 backbone for a Swin
Transformer, which can capture more complex feature relationships
through self-attention — potentially improving detection of
fine-grained damage patterns like hairline scratches.

Requires torchvision >= 0.15 for Swin backbone support.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models.base import BaseDetector
from models.registry import register_model
from utils.config import Config


@register_model("maskrcnn_swin")
class MaskRCNNSwinDamage(BaseDetector):
    """
    Mask R-CNN with Swin Transformer backbone.

    Uses torchvision's Swin-T backbone with FPN, plugged into
    the standard Mask R-CNN detection head.

    Config options (in model section):
        - num_classes: Number of classes including background
        - backbone: swin_t | swin_s | swin_b
        - pretrained: Use ImageNet pretrained backbone
        - min_size: Minimum image size
        - max_size: Maximum image size
    """

    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 2)
        pretrained = getattr(model_config, "pretrained", True)
        backbone_name = getattr(model_config, "backbone", "swin_t")
        min_size = getattr(model_config, "min_size", 800)
        max_size = getattr(model_config, "max_size", 1333)

        # Build Swin backbone with FPN
        backbone = self._build_swin_fpn_backbone(backbone_name, pretrained)

        # Build full Mask R-CNN with the Swin backbone
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=min_size,
            max_size=max_size,
        )

        return model

    def _build_swin_fpn_backbone(self, name: str, pretrained: bool):
        """Build a Swin Transformer backbone with FPN."""
        from torchvision.models import swin_t, swin_s, swin_b, Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
        from torchvision.models.detection.backbone_utils import BackboneWithFPN
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

        # Select the Swin variant
        swin_configs = {
            "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1 if pretrained else None),
            "swin_s": (swin_s, Swin_S_Weights.IMAGENET1K_V1 if pretrained else None),
            "swin_b": (swin_b, Swin_B_Weights.IMAGENET1K_V1 if pretrained else None),
        }

        if name not in swin_configs:
            raise ValueError(f"Unknown Swin variant: {name}. Available: {list(swin_configs.keys())}")

        model_fn, weights = swin_configs[name]
        swin = model_fn(weights=weights)

        # Extract feature layers from Swin's sequential features
        # Swin-T features output channels: [96, 192, 384, 768]
        backbone = torch.nn.Sequential(*list(swin.features.children()))

        # Wrap with FPN using intermediate feature extraction
        # Use IntermediateLayerGetter approach
        from torchvision.models._utils import IntermediateLayerGetter

        return_layers = {"1": "0", "3": "1", "5": "2", "7": "3"}
        in_channels_list = [96, 192, 384, 768]  # Swin-T channel sizes

        backbone_with_fpn = BackboneWithFPN(
            backbone=backbone,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )

        return backbone_with_fpn

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        return model(images, targets)

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
