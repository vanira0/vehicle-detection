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
import models.vehicle     # noqa: F401


def resize_with_aspect_ratio_and_pad(image, target_size=512):
    h, w = image.shape[:2]
    
    # Calculate the scaling factor to fit within the target square
    scaling_factor = target_size / max(h, w)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    
    # Resize the image using INTER_AREA (best for shrinking)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a solid black canvas of the target size
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate top-left offsets to center the image on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Paste the resized image onto the center of the canvas
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return padded_img


def pipeline_preprocess(image_path, target_size=512, max_size_kb=400):
    # 1. Load the original high-res image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image could not be loaded.")
        
    # 2. Resize and pad maintaining aspect ratio
    processed_img = resize_with_aspect_ratio_and_pad(img, target_size=target_size)
    
    # 3. Dynamically compress to target file size (< 400KB)
    quality = 95
    while quality > 10:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', processed_img, encode_param)
        
        size_kb = len(encimg) / 1024
        if size_kb <= max_size_kb:
            # Decode back to matrix to feed into your inference pipeline
            final_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            return final_img, quality, size_kb
            
        quality -= 5
        
    raise ValueError("Could not compress below target size even at lowest JPEG quality.")


def apply_clahe(image):
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image


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
    
    # Try to load class names from yaml_path if provided
    yaml_path = getattr(config.data, "yaml_path", None) if hasattr(config, "data") else None
    loaded_names = None
    if yaml_path and os.path.exists(yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            if "names" in data_yaml:
                if isinstance(data_yaml["names"], list):
                    loaded_names = {i: name for i, name in enumerate(data_yaml["names"])}
                elif isinstance(data_yaml["names"], dict):
                    loaded_names = data_yaml["names"]
    
    if "yolo" in model_name.lower():
        from ultralytics import YOLO
        yolo_model = YOLO(checkpoint_path)
        if loaded_names is not None:
            yolo_model.model.names = loaded_names
        model_wrapper._yolo_model = yolo_model
        model = yolo_model.model.to(device)
        model.eval()
    else:
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
            try:
                image, quality, size_kb = pipeline_preprocess(img_path, target_size=1024, max_size_kb=1024)
                logger.info(f"Compressed image to {size_kb:.2f}KB with quality {quality}")
            except Exception as e:
                logger.error(f"Preprocessing failed for {img_path}: {e}")
                continue
                
            image = apply_clahe(image)
                
            image_bgr = image.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image_rgb).to(device)
            
            is_classifier = getattr(config.model, "stage", "") in ["gatekeeper", "angle"]
            
            if is_classifier:
                results = model_wrapper.predict(model, image_tensor)
                pred_cls = results["predicted_class"]
                score = results["confidence"]
                
                if hasattr(model_wrapper, "_yolo_model") and model_wrapper._yolo_model is not None:
                    label_str = model_wrapper._yolo_model.names.get(int(pred_cls), f"Class {pred_cls}")
                else:
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
                if "yolo" in config.model.name.lower():
                    # For YOLO models, use Ultralytics API for inference
                    predictions = model_wrapper._yolo_model.predict(image_bgr, verbose=False, conf=args.conf_thresh)
                else:
                    with torch.no_grad():
                        predictions = model([image_tensor])
                    
                results = model_wrapper.post_process(predictions, confidence_threshold=args.conf_thresh)[0]
                
                if hasattr(model_wrapper, "select_target_vehicle"):
                    target_idx = model_wrapper.select_target_vehicle(results, image_bgr.shape)
                    if target_idx is not None:
                        results["boxes"] = np.array([results["boxes"][target_idx]])
                        results["labels"] = np.array([results["labels"][target_idx]])
                        results["scores"] = np.array([results["scores"][target_idx]])
                        if "masks" in results and results["masks"] is not None:
                            results["masks"] = np.array([results["masks"][target_idx]])
                    else:
                        results["boxes"] = np.zeros((0, 4))
                        results["labels"] = np.zeros((0,))
                        results["scores"] = np.zeros((0,))
                        if "masks" in results and results["masks"] is not None:
                            results["masks"] = np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]))

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
                    cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), color, 3)
                    
                    # Draw label
                    if hasattr(model_wrapper, "_yolo_model") and model_wrapper._yolo_model is not None:
                        label_str = model_wrapper._yolo_model.names.get(label, f"Class {label}")
                    else:
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3
                    )
                    
                    # Draw mask if available
                    # if masks is not None:
                    #     mask = masks[i]
                    #     colored_mask = np.zeros_like(image_bgr)
                    #     colored_mask[mask > 0] = color
                    #     alpha = 0.5
                    #     mask_indices = mask > 0
                    #     image_bgr[mask_indices] = cv2.addWeighted(
                    #         image_bgr, 1.0, colored_mask, alpha, 0
                    #     )[mask_indices]

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
