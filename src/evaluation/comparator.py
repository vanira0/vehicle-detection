"""
Experiment comparator for side-by-side comparison of multiple runs.

Loads metrics from experiment directories and generates formatted
comparison tables, supporting both terminal output and file export.
"""

import json
import os
from typing import Dict, List, Optional

import yaml


class ExperimentComparator:
    """
    Compare metrics across multiple experiment runs.

    Usage:
        comparator = ExperimentComparator(runs_dir="runs")
        comparator.add_experiment("damage_maskrcnn_v1")
        comparator.add_experiment("damage_swin_v1")
        comparator.print_comparison()
    """

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        self.experiments: Dict[str, Dict] = {}

    def add_experiment(self, name: str) -> None:
        """
        Load metrics from an experiment directory.

        Args:
            name: Experiment name (subdirectory under runs_dir).
        """
        exp_dir = os.path.join(self.runs_dir, name)
        if not os.path.isdir(exp_dir):
            print(f"Warning: Experiment directory not found: {exp_dir}")
            return

        # Load config
        config_path = os.path.join(exp_dir, "config.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        # Load metrics CSV (last row = final metrics)
        metrics = self._load_latest_metrics(exp_dir)

        # Count model parameters from config
        model_name = config.get("model", {}).get("name", "unknown")

        self.experiments[name] = {
            "config": config,
            "metrics": metrics,
            "model_name": model_name,
        }

    def _load_latest_metrics(self, exp_dir: str) -> Dict:
        """Load the latest metrics from the experiment."""
        metrics_file = os.path.join(exp_dir, "metrics", "metrics.csv")
        if not os.path.exists(metrics_file):
            return {}

        import csv

        with open(metrics_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {}

        # Return the last row (final epoch metrics)
        return {k: self._parse_value(v) for k, v in rows[-1].items()}

    @staticmethod
    def _parse_value(value: str):
        """Parse a string value to int, float, or keep as string."""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def print_comparison(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
    ) -> str:
        """
        Print a formatted comparison table.

        Args:
            metrics: List of metric names to include. If None, include all.
            sort_by: Metric name to sort experiments by (descending).

        Returns:
            Formatted table string.
        """
        if not self.experiments:
            return "No experiments to compare."

        try:
            from tabulate import tabulate
        except ImportError:
            return self._simple_comparison(metrics)

        # Determine columns
        if metrics is None:
            # Collect all metric keys across experiments
            all_keys = set()
            for exp_data in self.experiments.values():
                all_keys.update(exp_data["metrics"].keys())
            metrics = sorted(all_keys)

        # Build table rows
        headers = ["Experiment", "Model"] + metrics
        rows = []

        for name, data in self.experiments.items():
            row = [name, data["model_name"]]
            for m in metrics:
                val = data["metrics"].get(m, "—")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            rows.append(row)

        # Sort if requested
        if sort_by and sort_by in metrics:
            col_idx = metrics.index(sort_by) + 2  # +2 for name and model cols
            rows.sort(
                key=lambda r: float(r[col_idx]) if r[col_idx] != "—" else -1,
                reverse=True,
            )

        table = tabulate(rows, headers=headers, tablefmt="grid")
        print(table)
        return table

    def _simple_comparison(self, metrics: Optional[List[str]] = None) -> str:
        """Fallback comparison without tabulate."""
        lines = []
        for name, data in self.experiments.items():
            lines.append(f"\n=== {name} (model: {data['model_name']}) ===")
            for k, v in data["metrics"].items():
                if metrics is None or k in metrics:
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")
        result = "\n".join(lines)
        print(result)
        return result

    def export_comparison(self, output_path: str) -> None:
        """Export comparison data to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.experiments, f, indent=2, default=str)
