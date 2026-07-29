import argparse
import json
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import models.gatekeeper
import models.angle
import models.damage
import models.parts

from inference.pipeline import VehicleDamagePipeline
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run vehicle damage detection pipeline on a directory of images")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to directory containing input images")
    parser.add_argument("--gatekeeper", type=str, required=True, help="Gatekeeper checkpoint")
    parser.add_argument("--damage", type=str, required=True, help="Damage model checkpoint")
    parser.add_argument("--parts", type=str, required=True, help="Parts model checkpoint")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for mapping")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for JSON results")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("bulk_infer")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build pipeline once (this is efficient)
    logger.info("Loading models...")
    pipeline = VehicleDamagePipeline(
        gatekeeper_checkpoint=args.gatekeeper,
        damage_checkpoint=args.damage,
        parts_checkpoint=args.parts,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Find all images (jpg, jpeg, png)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    
    logger.info(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        logger.info(f"Processing: {img_path}")
        try:
            result = pipeline(img_path)
            
            # Save results
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.json")
            
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")


if __name__ == "__main__":
    main()
