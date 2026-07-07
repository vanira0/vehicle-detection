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
from torchvision.models.detection.roi_heads import project_masks_on_boxes
from torchvision.models.detection.roi_heads import maskrcnn_loss as default_maskrcnn_loss

def bce_dice_mask_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Custom mask loss that combines standard CrossEntropy/BCE with Dice Loss.
    """
    # Standard loss from torchvision
    bce_loss = default_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
    
    if mask_logits.numel() == 0:
        return bce_loss
        
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    
    if labels.numel() == 0:
        return bce_loss
        
    # Get mask logits corresponding to the target labels
    idx = torch.arange(labels.shape[0], device=labels.device)
    mask_logits_pos = mask_logits[idx, labels]
    
    # Dice computation
    inputs = mask_logits_pos.sigmoid().flatten(1)
    targets = mask_targets.flatten(1)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    dice_loss = 1 - (numerator + 1) / (denominator + 1)
    
    return 0.5 * bce_loss + 0.5 * dice_loss.mean()

from .base import BaseDetector
from utils.config import Config
from .registry import register_model


@register_model("maskrcnn_seg")
class MaskRCNNSegmentationWrapper(BaseDetector):
    """
    Wrapper for PyTorch's native Mask R-CNN.
    """
    
    def __init__(self):
        self.config = None
        
    def build(self, model_config: Config) -> nn.Module:
        """
        Construct the model.
        """
        self.config = model_config
        num_classes = getattr(model_config, "num_classes", 2)
        mask_loss_type = getattr(model_config, "mask_loss_type", "bce")
        
        # Monkey-patch the mask loss if a custom one is requested
        import torchvision.models.detection.roi_heads as roi_heads
        if mask_loss_type == "bce_dice":
            roi_heads.maskrcnn_loss = bce_dice_mask_loss
        else:
            roi_heads.maskrcnn_loss = default_maskrcnn_loss
        
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
        
        # Apply mask loss multiplier if specified
        if self.config is not None:
            mask_weight = getattr(self.config, "mask_loss_weight", 1.0)
            if "loss_mask" in loss_dict and mask_weight != 1.0:
                loss_dict["loss_mask"] = loss_dict["loss_mask"] * mask_weight
                
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
