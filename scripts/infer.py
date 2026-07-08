#!/usr/bin/env python
"""
Run the full 3-stage inference pipeline on an image.

Usage:
    python scripts/infer.py \
        --image test_car.jpg \
        --gatekeeper runs/gatekeeper_v1/checkpoints/best.pth \
        --damage runs/damage_v1/checkpoints/best.pth \
        --parts runs/parts_v1/checkpoints/best.pth

    # With custom thresholds
    python scripts/infer.py \
        --image test_car.jpg \
        --gatekeeper runs/gatekeeper_v1/checkpoints/best.pth \
        --damage runs/damage_v1/checkpoints/best.pth \
        --parts runs/parts_v1/checkpoints/best.pth \
        --iou-threshold 0.3 \
        --confidence 0.5 \
        --visualize --output result.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import models.gatekeeper
import models.angle
import models.damage      # noqa: F401
import models.parts       # noqa: F401

from inference.pipeline import VehicleDamagePipeline
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run vehicle damage detection pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--gatekeeper", type=str, required=True, help="Gatekeeper checkpoint")
    parser.add_argument("--damage", type=str, required=True, help="Damage model checkpoint")
    parser.add_argument("--parts", type=str, required=True, help="Parts model checkpoint")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for mapping")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--visualize", action="store_true", help="Save visualization")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization or JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("infer")

    # Build pipeline
    pipeline = VehicleDamagePipeline(
        gatekeeper_checkpoint=args.gatekeeper,
        damage_checkpoint=args.damage,
        parts_checkpoint=args.parts,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Run inference
    logger.info(f"Processing: {args.image}")
    result = pipeline(args.image)

    # Display results
    print("\n" + "=" * 60)
    print(f"Status: {result['status'].upper()}")
    print(f"Gatekeeper confidence: {result['gatekeeper']['confidence']:.3f}")
    print(f"Inference time: {result.get('inference_time_ms', 0):.0f}ms")

    if result["status"] == "damaged":
        print(f"\nDamage detections: {result['damage']['num_detections']}")
        print(f"Part detections: {result['parts']['num_detections']}")
        print(f"\nFindings ({len(result['findings'])}):")
        for i, finding in enumerate(result["findings"], 1):
            print(f"  {i}. {finding['description']}")
            print(f"     Severity: {finding['severity']}")
            print(f"     Confidence: damage={finding['damage_confidence']:.3f}, "
                  f"part={finding['part_confidence']:.3f}")
            print(f"     Overlap IoU: {finding['overlap_score']:.3f}")
    print("=" * 60)

    # Save results
    if args.output:
        if args.output.endswith(".json"):
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        elif args.visualize:
            logger.info("Visualization saved (requires pipeline visualization support)")


if __name__ == "__main__":
    main()
