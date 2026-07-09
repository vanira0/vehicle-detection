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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference.configurable_pipeline import ConfigurablePipeline
from utils.logger import setup_logger


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


def append_to_csv(model_name: str, image_name: str, context_data: dict, output_dir: str):
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
                    writer.writerow(["image_name", "predicted_class", "confidence"])
                elif "is_damaged" in context_data:
                    writer.writerow(["image_name", "is_damaged", "confidence"])
            
            if "predicted_class" in context_data:
                writer.writerow([image_name, context_data["predicted_class"], context_data.get("confidence", 0.0)])
            elif "is_damaged" in context_data:
                writer.writerow([image_name, context_data["is_damaged"], context_data.get("confidence", 0.0)])
        return

    # Detection/Segmentation (e.g. parts, damage)
    if "boxes" in context_data and "labels" in context_data:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_empty:
                writer.writerow(["image_name", "class_id", "score", "box_x1", "box_y1", "box_x2", "box_y2"])
            
            boxes = context_data["boxes"]
            labels = context_data["labels"]
            scores = context_data.get("scores", [])
            
            for i in range(len(labels)):
                box = boxes[i]
                label = int(labels[i])
                score = float(scores[i]) if i < len(scores) else 1.0
                writer.writerow([image_name, label, score, box[0], box[1], box[2], box[3]])
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
                result = pipeline(image_path)
                
                # Extract context for CSV export
                context = result.pop("_context", {})
                
                # Update total time
                total_time += result.get("inference_time_ms", 0)
                
                # Export to CSVs
                for model_cfg in pipeline.config.get("models", []):
                    name = model_cfg["name"]
                    if name in context:
                        append_to_csv(name, image_name, context[name], args.output_dir)
                
                # Write to JSONL
                jsonl_file.write(json.dumps(result, default=default_serializer) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing {image_name}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Processed {len(image_paths)} images in {total_time:.0f}ms")
    print(f"Average time per image: {total_time/len(image_paths):.0f}ms")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
