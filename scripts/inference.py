import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.registry import get_model
from utils.config import Config

# Import model packages to trigger registration
import models.gatekeeper  # noqa: F401
import models.damage      # noqa: F401
import models.parts       # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model on an image")
    parser.add_argument("--config", type=str, required=True, help="configs/damage/maskrcnn_resnet50.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="runs/damage_maskrcnn_resnet50_v1/checkpoints/best.pth")
    parser.add_argument("--image", type=str, required=True, help="input_image")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output.jpg", help="test_output_image")
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    # Load config
    config = Config.from_file(config_path)
    
    # Get model wrapper from registry
    model_name = config.model.name
    print(f"Loading model strategy: {model_name}")
    model_wrapper = get_model(model_name)()
    
    # Build model using config
    model = model_wrapper.build(config.model).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, model_wrapper, config


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model, model_wrapper, config = load_model(args.config, args.checkpoint, device)
    
    # Load and prepare image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image at {args.image}")
        return
        
    # Keep original for drawing
    image_bgr = image.copy()
    
    # Convert BGR to RGB for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize (0 to 1)
    # Note: Torchvision Mask R-CNN expects float tensors in [0, 1]
    image_tensor = F.to_tensor(image_rgb).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model([image_tensor])
        
    # Post-process predictions
    print(f"Filtering with confidence threshold: {args.conf_thresh}")
    results = model_wrapper.post_process(predictions, confidence_threshold=args.conf_thresh)[0]
    
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]
    
    print(f"Found {len(boxes)} objects!")
    
    # Optionally get masks if it's a segmentation model
    masks = results.get("masks", None)
    
    # Colors for different classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    # Draw results on the image
    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        label = int(labels[i])
        score = scores[i]
        
        color = [int(c) for c in colors[label]]
        
        # Draw box
        cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label
        class_names = config.data.get("class_names", []) if hasattr(config, "data") else []
        if class_names and label < len(class_names):
            label_str = class_names[label]
        else:
            label_str = f"Class {label}"
        label_text = f"{label_str} ({score:.2f})"
        cv2.putText(
            image_bgr, label_text, (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        
        # Draw mask if available
        if masks is not None:
            mask = masks[i]
            # Create a colored overlay for the mask
            colored_mask = np.zeros_like(image_bgr)
            colored_mask[mask > 0] = color
            
            # Blend the mask with the image
            alpha = 0.5
            mask_indices = mask > 0
            image_bgr[mask_indices] = cv2.addWeighted(
                image_bgr, 1.0, colored_mask, alpha, 0
            )[mask_indices]

    # Save output
    cv2.imwrite(args.output, image_bgr)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
