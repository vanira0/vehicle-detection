import argparse
import os
import sys
import glob
import csv

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.registry import get_model
from utils.config import Config
from utils.logger import setup_logger

# Import model packages to trigger registration
import models.gatekeeper
import models.angle
import models.damage      # noqa: F401
import models.parts       # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained individual model on a directory of images")
    parser.add_argument("--config", type=str, required=True, help="Path to model config (e.g., configs/damage/maskrcnn_resnet50.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output visualizations")
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    # Load config
    config = Config.from_file(config_path)
    
    # Get model wrapper from registry
    model_name = config.model.name
    model_wrapper = get_model(model_name)()
    
    # Build model using config
    model = model_wrapper.build(config.model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, model_wrapper, config


def main():
    args = parse_args()
    logger = setup_logger("bulk_inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model once
    logger.info(f"Loading model from {args.checkpoint}...")
    model, model_wrapper, config = load_model(args.config, args.checkpoint, device)
    
    # Find all images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
    logger.info(f"Found {len(image_paths)} images to process.")
    
    # Random colors for drawing bounding boxes/masks
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image_name", "classes", "scores", "boxes"])
    
    for img_path in image_paths:
        logger.info(f"Processing: {img_path}")
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.error(f"Could not load image at {img_path}")
                continue
                
            image_bgr = image.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image_rgb).to(device)
            
            is_classifier = getattr(config.model, "stage", "") in ["gatekeeper", "angle"]
            
            if is_classifier:
                results = model_wrapper.predict(model, image_tensor)
                pred_cls = results["predicted_class"]
                score = results["confidence"]
                
                class_names = config.data.get("class_names", []) if hasattr(config, "data") else []
                if class_names and pred_cls < len(class_names):
                    label_str = class_names[pred_cls]
                else:
                    label_str = f"Class {pred_cls}"
                label_text = f"{label_str} ({score:.2f})"
                
                img_classes = [label_str]
                img_scores = [f"{score:.4f}"]
                img_boxes = ["[]"]
                
                cv2.putText(
                    image_bgr, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )
            else:
                with torch.no_grad():
                    predictions = model([image_tensor])
                    
                results = model_wrapper.post_process(predictions, confidence_threshold=args.conf_thresh)[0]
                
                boxes = results["boxes"]
                labels = results["labels"]
                scores = results["scores"]
                masks = results.get("masks", None)
                
                # Draw results on the image
                img_classes = []
                img_scores = []
                img_boxes = []
                
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
                    
                    img_classes.append(label_str)
                    img_scores.append(f"{score:.4f}")
                    img_boxes.append(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
                    
                    cv2.putText(
                        image_bgr, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                    
                    # Draw mask if available
                    if masks is not None:
                        mask = masks[i]
                        colored_mask = np.zeros_like(image_bgr)
                        colored_mask[mask > 0] = color
                        alpha = 0.5
                        mask_indices = mask > 0
                        image_bgr[mask_indices] = cv2.addWeighted(
                            image_bgr, 1.0, colored_mask, alpha, 0
                        )[mask_indices]

            # Save output image
            base_name = os.path.basename(img_path)
            
            with open(csv_path, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    base_name,
                    ", ".join(img_classes),
                    ", ".join(img_scores),
                    ", ".join(img_boxes)
                ])
                
            output_path = os.path.join(args.output_dir, base_name)
            cv2.imwrite(output_path, image_bgr)
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")

    logger.info("Bulk inference completed!")

if __name__ == "__main__":
    main()
