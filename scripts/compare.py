#!/usr/bin/env python
"""
Compare metrics across multiple experiment runs.

Usage:
    python scripts/compare.py --runs damage_maskrcnn_v1 damage_swin_v1 damage_yolov8_v1
    python scripts/compare.py --runs damage_maskrcnn_v1 damage_swin_v1 --sort-by mAP_50_95
    python scripts/compare.py --runs-dir runs --all --export comparison.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.comparator import ExperimentComparator


def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument(
        "--runs", nargs="*", default=[],
        help="Names of experiment runs to compare",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Compare all experiments in runs-dir",
    )
    parser.add_argument(
        "--metrics", nargs="*", default=None,
        help="Specific metrics to compare",
    )
    parser.add_argument(
        "--sort-by", type=str, default=None,
        help="Metric to sort experiments by",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Path to export comparison as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    comparator = ExperimentComparator(runs_dir=args.runs_dir)

    # Add experiments
    if args.all:
        # Auto-discover all experiments
        if os.path.isdir(args.runs_dir):
            for name in sorted(os.listdir(args.runs_dir)):
                exp_path = os.path.join(args.runs_dir, name)
                if os.path.isdir(exp_path):
                    comparator.add_experiment(name)
    else:
        for name in args.runs:
            comparator.add_experiment(name)

    if not comparator.experiments:
        print("No experiments found to compare.")
        print(f"Run experiments first, or check --runs-dir: {args.runs_dir}")
        return

    # Print comparison
    comparator.print_comparison(
        metrics=args.metrics,
        sort_by=args.sort_by,
    )

    # Export if requested
    if args.export:
        comparator.export_comparison(args.export)
        print(f"\nComparison exported to: {args.export}")


if __name__ == "__main__":
    main()
