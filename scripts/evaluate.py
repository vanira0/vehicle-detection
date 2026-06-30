#!/usr/bin/env python
"""
Evaluate a trained model checkpoint on a test dataset.

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/damage_maskrcnn_v1/checkpoints/best.pth \
        --data data/processed/test \
        --annotations data/annotations/test.json \
        --output results/damage_maskrcnn_v1_eval.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader

from data.augmentations import build_augmentation_pipeline
from data.dataset import ClassificationDataset, COCOSegmentationDataset, collate_fn
from evaluation.evaluator import Evaluator
from models.registry import get_model

import models.gatekeeper  # noqa: F401
import models.damage      # noqa: F401
import models.parts       # noqa: F401

from utils.config import Config
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to test data directory")
    parser.add_argument("--annotations", type=str, default=None, help="Path to COCO annotations JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("evaluate")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = Config.from_dict(checkpoint["config"])

    # Build model
    model_name = config.model.name
    model_wrapper = get_model(model_name)()
    model = model_wrapper.build(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Model: {model_name} | Stage: {config.model.stage}")
    logger.info(f"Loaded from epoch: {checkpoint.get('epoch', '?')}")

    # Build test data loader
    stage = config.model.stage
    data_root = args.data or os.path.join(config.data.root, "test")
    val_transform = build_augmentation_pipeline(config, is_train=False)

    if stage == "gatekeeper":
        test_dataset = ClassificationDataset(root=data_root, transform=val_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    else:
        ann_file = args.annotations or os.path.join(
            getattr(config.data, "annotations_dir", "data/annotations"), "test.json"
        )
        test_dataset = COCOSegmentationDataset(
            root=data_root, annotation_file=ann_file, transform=val_transform,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=collate_fn,
        )

    logger.info(f"Test dataset: {len(test_dataset)} images")

    # Run evaluation
    model_type = "classification" if stage == "gatekeeper" else "detection"
    evaluator = Evaluator(
        model_type=model_type,
        num_classes=getattr(config.model, "num_classes", 2),
        confidence_threshold=args.confidence,
    )

    metrics = evaluator.evaluate(model, test_loader, device)

    # Print results
    logger.info("=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
