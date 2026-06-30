"""
YAML configuration loader with inheritance and CLI override support.

Features:
    - Base config inheritance via `_base_` key
    - Recursive dict merging (child overrides parent)
    - CLI overrides via `--set key.subkey=value`
    - Dot-notation attribute access (config.training.optimizer.lr)
    - Frozen config snapshot saved to experiment directory
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Config:
    """
    Hierarchical configuration object with dot-notation access.

    Usage:
        config = Config.from_file("configs/damage/maskrcnn_resnet50.yaml")
        config = Config.from_file("configs/damage/maskrcnn_resnet50.yaml",
                                  overrides=["training.optimizer.lr=0.001"])
        print(config.training.optimizer.lr)  # 0.001
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_file(cls, path: str, overrides: Optional[List[str]] = None) -> "Config":
        """
        Load a YAML config file, resolving `_base_` inheritance chain,
        then applying CLI overrides.

        Args:
            path: Path to the YAML config file.
            overrides: List of "key.subkey=value" strings for CLI overrides.

        Returns:
            Config object with dot-notation access.
        """
        raw = cls._load_with_inheritance(path)
        if overrides:
            raw = cls._apply_overrides(raw, overrides)
        return cls(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config from a plain dictionary."""
        return cls(copy.deepcopy(data))

    @classmethod
    def _load_with_inheritance(cls, path: str) -> Dict[str, Any]:
        """
        Recursively load YAML configs following `_base_` references.
        Child values override parent values.
        """
        path = Path(path).resolve()
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        base_path = data.pop("_base_", None)
        if base_path:
            # Resolve relative to the current config file's directory
            base_full_path = (path.parent / base_path).resolve()
            if not base_full_path.exists():
                raise FileNotFoundError(
                    f"Base config not found: {base_full_path} "
                    f"(referenced from {path})"
                )
            base_data = cls._load_with_inheritance(str(base_full_path))
            data = cls._deep_merge(base_data, data)

        return data

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge `override` into `base`.
        Override values take precedence. Dicts are merged recursively;
        non-dict values are replaced entirely.
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _apply_overrides(data: Dict, overrides: List[str]) -> Dict:
        """
        Apply CLI overrides in the form "key.subkey=value".
        Values are parsed as YAML to support int, float, bool, null, etc.
        """
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid override format: '{override}'. "
                    f"Expected 'key.subkey=value'."
                )
            key_path, value_str = override.split("=", 1)
            keys = key_path.strip().split(".")

            # Parse value as YAML (handles int, float, bool, null, lists)
            value = yaml.safe_load(value_str.strip())

            # Walk down the dict and set the value
            d = data
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value

        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config object back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = copy.deepcopy(value)
        return result

    def save(self, path: str) -> None:
        """Save config snapshot to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value by dot-separated key path with a default fallback.

        Args:
            key: Dot-separated key path, e.g. "training.optimizer.lr"
            default: Value to return if the key path doesn't exist.
        """
        keys = key.split(".")
        obj = self
        for k in keys:
            if isinstance(obj, Config) and hasattr(obj, k):
                obj = getattr(obj, k)
            elif isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
