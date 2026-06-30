#!/usr/bin/env python
"""
Export a trained model to ONNX or TorchScript format.

Usage:
    python scripts/export.py \
        --checkpoint runs/damage_v1/checkpoints/best.pth \
        --format onnx \
        --output exported/damage_model.onnx

    python scripts/export.py \
        --checkpoint runs/damage_v1/checkpoints/best.pth \
        --format torchscript \
        --output exported/damage_model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from models.registry import get_model

import models.gatekeeper  # noqa: F401
import models.damage      # noqa: F401
import models.parts       # noqa: F401

from inference.exporters import ONNXExporter, TorchScriptExporter
from utils.config import Config
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to deployment format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 1024, 1024],
                        help="Input tensor shape: batch channels height width")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("export")

    # Load checkpoint
    device = torch.device("cpu")  # Export on CPU
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = Config.from_dict(checkpoint["config"])

    # Build model
    model_name = config.model.name
    model_wrapper = get_model(model_name)()
    model = model_wrapper.build(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Model: {model_name}")
    logger.info(f"Export format: {args.format}")
    logger.info(f"Input size: {args.input_size}")

    # Export
    input_size = tuple(args.input_size)
    if args.format == "onnx":
        ONNXExporter.export(
            model, args.output,
            input_size=input_size,
            opset_version=args.opset,
        )
    else:
        TorchScriptExporter.export(
            model, args.output,
            input_size=input_size,
        )

    # Report file size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    logger.info(f"Exported model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
