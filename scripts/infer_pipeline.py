#!/usr/bin/env python
"""
Run the configurable inference pipeline on an image or a directory of images
and export results to CSV.

Usage:
    # Single image
    python scripts/infer_pipeline.py \
        --pipeline-config configs/pipeline/test_all.yaml \
        --input test_car.jpg \
        --output-dir results/

    # Directory of images
    python scripts/infer_pipeline.py \
        --pipeline-config configs/pipeline/test_all.yaml \
        --input path/to/images_dir/ \
        --output-dir results/
"""

import argparse
import csv
import json
import os
import sys
import glob
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference.configurable_pipeline import ConfigurablePipeline
from utils.logger import setup_logger
from data.preprocessing import resize_image
from utils.visualization import draw_bboxes
import cv2


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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image




def _compute_box_iou(box1, box2) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)




def parse_args():
    parser = argparse.ArgumentParser(description="Run configurable vehicle damage detection pipeline")
    parser.add_argument("--pipeline-config", type=str, required=True, help="Path to pipeline YAML config")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSVs and JSON")
    return parser.parse_args()


def init_csv_files(models, output_dir: str):
    """Initialize CSV files with headers."""
    for model_cfg in models:
        name = model_cfg["name"]
        csv_path = os.path.join(output_dir, f"{name}_predictions.csv")
        
        # We can't know the exact output format just from the config, but we can write headers on first append
        # So we'll just clear the files here.
        with open(csv_path, "w", newline="") as f:
            pass


def append_to_csv(model_name: str, image_name: str, context_data: dict, output_dir: str, class_names: list = None):
    """Append model predictions to CSV for a single image."""
    csv_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    
    # Check if file is empty to write header
    is_empty = os.path.getsize(csv_path) == 0 if os.path.exists(csv_path) else True

    # Classification (e.g. angle, gatekeeper)
    if "predicted_class" in context_data or "is_damaged" in context_data:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_empty:
                if "predicted_class" in context_data:
                    writer.writerow(["image_name", "predicted_class", "class_name", "confidence"])
                elif "is_damaged" in context_data:
                    writer.writerow(["image_name", "is_damaged", "confidence"])
            
            if "predicted_class" in context_data:
                pred_class = context_data["predicted_class"]
                class_name = ""
                if class_names is not None:
                    try:
                        idx = int(pred_class)
                        if 0 <= idx < len(class_names):
                            class_name = class_names[idx]
                    except (ValueError, TypeError):
                        pass
                writer.writerow([image_name, pred_class, class_name, context_data.get("confidence", 0.0)])
            elif "is_damaged" in context_data:
                writer.writerow([image_name, context_data["is_damaged"], context_data.get("confidence", 0.0)])
        return

    # Detection/Segmentation (e.g. parts, damage)
    if "boxes" in context_data and "labels" in context_data:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_empty:
                writer.writerow(["image_name", "class_id", "class_name", "score", "box_x1", "box_y1", "box_x2", "box_y2"])
            
            boxes = context_data["boxes"]
            labels = context_data["labels"]
            scores = context_data.get("scores", [])
            
            for i in range(len(labels)):
                box = boxes[i]
                label = int(labels[i])
                class_name = ""
                if class_names is not None and 0 <= label < len(class_names):
                    class_name = class_names[label]
                score = float(scores[i]) if i < len(scores) else 1.0
                writer.writerow([image_name, label, class_name, score, box[0], box[1], box[2], box[3]])
        return


def main():
    args = parse_args()
    logger = setup_logger("infer_pipeline")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine inputs
    image_paths = []
    if os.path.isdir(args.input):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in os.listdir(args.input):
            if os.path.splitext(f.lower())[1] in valid_exts:
                image_paths.append(os.path.join(args.input, f))
        logger.info(f"Found {len(image_paths)} images in directory {args.input}")
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        logger.error(f"Input path {args.input} is not a valid file or directory.")
        sys.exit(1)

    if not image_paths:
        logger.warning("No images found to process.")
        return

    # Build pipeline
    logger.info(f"Loading pipeline from: {args.pipeline_config}")
    pipeline = ConfigurablePipeline(config_path=args.pipeline_config)
    
    # Initialize CSV files
    init_csv_files(pipeline.config.get("models", []), args.output_dir)

    jsonl_path = os.path.join(args.output_dir, "pipeline_results.jsonl")
    
    def default_serializer(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)
    
    # Clear JSONL file
    with open(jsonl_path, "w") as f:
        pass

    logger.info(f"Starting inference on {len(image_paths)} images...")
    
    total_time = 0
    with open(jsonl_path, "a") as jsonl_file:
        for idx, image_path in enumerate(image_paths, 1):
            image_name = os.path.basename(image_path)
            logger.info(f"[{idx}/{len(image_paths)}] Processing: {image_name}")
            
            try:
                # Preprocess and Apply CLAHE
                try:
                    img, quality, size_kb = pipeline_preprocess(image_path, target_size=1024, max_size_kb=1024)
                    logger.info(f"Compressed image to {size_kb:.2f}KB with quality {quality}")
                except Exception as e:
                    logger.error(f"Preprocessing failed for {image_name}: {e}")
                    img = None

                # apply clahe or not
                if img is not None:
                    clahe_img = apply_clahe(img)
                    tmp_path = os.path.join(args.output_dir, "tmp_clahe.jpg")
                    cv2.imwrite(tmp_path, clahe_img)
                    pipeline_input = tmp_path
                else:
                    pipeline_input = image_path
                
                # pipeline_input = image_path
                    
                result = pipeline(pipeline_input)
                result["image_path"] = image_path  # restore original path
                
                # Extract context for CSV export
                context = result.pop("_context", {})
                
                # Update total time
                total_time += result.get("inference_time_ms", 0)
                
                # Export to CSVs
                for m in pipeline.models:
                    name = m["name"]
                    if name in context:
                        class_names = m.get("config", {}).get("data.class_names")
                        append_to_csv(name, image_name, context[name], args.output_dir, class_names)
                
                # Write to JSONL
                jsonl_file.write(json.dumps(result, default=default_serializer) + "\n")
                
                # Visualize parts boxes and angle
                if "image_rgb" in context:
                    img_bgr = cv2.cvtColor(context["image_rgb"], cv2.COLOR_RGB2BGR)
                    
                    # Draw angle
                    if "angle" in result and "predicted_class" in result["angle"]:
                        pred_class = result['angle']['predicted_class']
                        angle_classes = None
                        for m in pipeline.models:
                            if m["name"] == "angle":
                                angle_classes = m.get("config", {}).get("data.class_names")
                                break
                        
                        class_name = str(pred_class)
                        if angle_classes is not None:
                            try:
                                pred_class_idx = int(pred_class)
                                if 0 <= pred_class_idx < len(angle_classes):
                                    class_name = angle_classes[pred_class_idx]
                            except (ValueError, TypeError):
                                pass
                                
                        angle_text = f"Angle: {class_name}"
                        cv2.putText(img_bgr, angle_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Extract damaged part indices from orchestrator findings
                    damaged_part_indices = set()
                    findings = result.get("findings", [])
                    for finding in findings:
                        if "part_index" in finding:
                            damaged_part_indices.add(finding["part_index"])


                    # Supplement with bounding-box overlap between damage and parts
                    # (covers cases where masks are unavailable or IoU threshold not met)
                    damage_ctx = context.get("damage", {})
                    parts_ctx = context.get("parts", {})
                    d_boxes = damage_ctx.get("boxes") if damage_ctx else None
                    p_boxes = parts_ctx.get("boxes") if parts_ctx else None
                    if (d_boxes is not None and p_boxes is not None
                            and len(d_boxes) > 0 and len(p_boxes) > 0):
                        for pi, p_box in enumerate(p_boxes):
                            for d_box in d_boxes:
                                if _compute_box_iou(p_box, d_box) > 0.05:
                                    damaged_part_indices.add(pi)
                                    break


                    # Draw parts boxes (only for damaged parts)
                    if "parts" in context:
                        parts_context = context["parts"]
                        boxes = parts_context.get("boxes")
                        labels = parts_context.get("labels")
                        scores = parts_context.get("scores")
                        
                        part_classes = None
                        for m in pipeline.models:
                            if m["name"] == "parts":
                                if hasattr(m["wrapper"], "_yolo_model"):
                                    part_classes = m["wrapper"]._yolo_model.names
                                else:
                                    part_classes = m.get("config", {}).get("data.class_names")
                                break
                                
                        if boxes is not None and len(boxes) > 0:
                            filtered_boxes = []
                            filtered_labels = []
                            filtered_scores = []
                            
                            for i in range(len(boxes)):
                                if i in damaged_part_indices:
                                    filtered_boxes.append(boxes[i])
                                    
                                    name = str(labels[i])
                                    if part_classes is not None:
                                        if isinstance(part_classes, dict) and int(labels[i]) in part_classes:
                                            name = part_classes[int(labels[i])]
                                        elif isinstance(part_classes, list) and 0 <= int(labels[i]) < len(part_classes):
                                            name = part_classes[int(labels[i])]
                                    filtered_labels.append(name)
                                    
                                    if scores is not None and i < len(scores):
                                        filtered_scores.append(scores[i])
                                        
                            if len(filtered_boxes) > 0:
                                img_bgr = draw_bboxes(img_bgr, np.array(filtered_boxes), filtered_labels, np.array(filtered_scores) if filtered_scores else None, score_threshold=pipeline.confidence_threshold, label_position="inside_bottom_left")
                                
                    # Draw damage boxes
                    if "damage" in context:
                        damage_context = context["damage"]
                        boxes = damage_context.get("boxes")
                        labels = damage_context.get("labels")
                        scores = damage_context.get("scores")
                        
                        damage_classes = None
                        for m in pipeline.models:
                            if m["name"] == "damage":
                                if hasattr(m["wrapper"], "_yolo_model"):
                                    damage_classes = m["wrapper"]._yolo_model.names
                                else:
                                    damage_classes = m.get("config", {}).get("data.class_names")
                                break
                                
                        if boxes is not None and len(boxes) > 0:
                            damage_labels_str = []
                            for l in labels:
                                name = str(l)
                                if damage_classes is not None:
                                    if isinstance(damage_classes, dict) and int(l) in damage_classes:
                                        name = f"Damage: {damage_classes[int(l)]}"
                                    elif isinstance(damage_classes, list) and 0 <= int(l) < len(damage_classes):
                                        name = f"Damage: {damage_classes[int(l)]}"
                                    else:
                                        name = f"Damage: {name}"
                                else:
                                    name = f"Damage: {name}"
                                damage_labels_str.append(name)
                                
                            img_bgr = draw_bboxes(img_bgr, np.array(boxes), damage_labels_str, scores, score_threshold=pipeline.confidence_threshold)
                            
                    annotated_path = os.path.join(args.output_dir, f"annotated_{image_name}")
                    cv2.imwrite(annotated_path, img_bgr)
                
            except Exception as e:
                logger.error(f"Error processing {image_name}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Processed {len(image_paths)} images in {total_time:.0f}ms")
    print(f"Average time per image: {total_time/len(image_paths):.0f}ms")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
