import numpy as np
from typing import Any, Dict, Optional
from models.yolo_segmentation import YOLO11SegmentationWrapper
from models.registry import register_model

@register_model("yolo11_vehicle_seg")
class YOLO11VehicleSegWrapper(YOLO11SegmentationWrapper):
    """Single-class vehicle segmentation for target isolation."""

    def select_target_vehicle(self, predictions: Dict[str, np.ndarray], image_shape: tuple) -> Optional[int]:
        """
        Pick the target vehicle from multiple detections using a combination of
        largest bounding box area and most central vehicle.
        
        Args:
            predictions: Dict with 'boxes', 'masks', 'scores', 'labels'
            image_shape: Tuple (H, W, C) or (H, W)
            
        Returns:
            int: Index of the selected vehicle, or None if no vehicles found.
        """
        boxes = predictions.get("boxes", [])
        if len(boxes) == 0:
            return None
            
        img_h, img_w = image_shape[:2]
        center_x, center_y = img_w / 2.0, img_h / 2.0
        
        best_idx = 0
        best_score = -float('inf')
        
        # Calculate max possible area and max possible distance to normalize
        max_area = img_w * img_h
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0
            dist_to_center = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
            
            # Normalize area (higher is better) and distance (lower is better)
            norm_area = area / max_area
            norm_dist = 1.0 - (dist_to_center / max_dist) # invert so higher is better
            
            # Combine scores (giving equal weight for now)
            # You can tune weights if needed: e.g., 0.6 * norm_area + 0.4 * norm_dist
            combined_score = 0.5 * norm_area + 0.5 * norm_dist
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
                
        return best_idx
        
    def extract_vehicle_roi(self, image: np.ndarray, predictions: Dict[str, Any], target_idx: int) -> np.ndarray:
        """
        Crop and mask the image to isolate the vehicle (crop+mask strategy).
        
        Args:
            image: Original image (H, W, 3)
            predictions: Dict containing 'boxes' and 'masks'
            target_idx: Index of the target vehicle
            
        Returns:
            np.ndarray: Cropped and masked vehicle image.
        """
        box = predictions["boxes"][target_idx]
        mask = predictions["masks"][target_idx]
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)
        
        # Expand mask to 3 channels for element-wise multiplication
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Mask the original image (everything outside mask becomes 0)
        masked_image = image * mask_3c
        
        # Crop to the bounding box
        x1, y1, x2, y2 = map(int, box)
        
        # Add some padding (optional, but good for context)
        padding = 10
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped_masked_image = masked_image[y1:y2, x1:x2]
        
        return cropped_masked_image
