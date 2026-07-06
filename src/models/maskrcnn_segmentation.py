"""
Mask R-CNN Segmentation Model Wrapper.

Wraps torchvision's Mask R-CNN to fit into the standard pipeline.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .base import BaseDetector
from utils.config import Config
from .registry import register_model


@register_model("maskrcnn_seg")
class MaskRCNNSegmentationWrapper(BaseDetector):
    """
    Wrapper for PyTorch's native Mask R-CNN.
    """
    
    def __init__(self):
        pass
        
    def build(self, model_config: Config) -> nn.Module:
        """
        Construct the model.
        """
        num_classes = getattr(model_config, "num_classes", 2)
        
        # Load pre-trained model on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        return model

    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss using the native torchvision forward pass during training.
        Mask R-CNN returns a dictionary of losses.
        """
        loss_dict = model(images, targets)
        return loss_dict

    def post_process(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Converts torchvision Results objects into standard format.
        """
        results_list = []
        for result in predictions:
            scores = result["scores"].detach().cpu().numpy()
            
            # Filter by confidence
            mask_conf = scores >= confidence_threshold
            
            boxes = result["boxes"][mask_conf].detach().cpu().numpy()
            scores = scores[mask_conf]
            labels = result["labels"][mask_conf].detach().cpu().numpy()
            
            # masks are shape (N, 1, H, W) probabilities
            if "masks" in result and result["masks"].shape[0] > 0:
                masks_prob = result["masks"][mask_conf].squeeze(1).detach().cpu().numpy()
                masks = (masks_prob > 0.5).astype(np.uint8)
            else:
                # Fallback if no valid masks
                masks = np.zeros((len(boxes), 0, 0), dtype=np.uint8)
                
            processed = {
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "masks": masks
            }
            results_list.append(processed)
            
        return results_list
