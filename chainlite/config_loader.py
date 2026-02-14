"""Config loading helpers for ChainLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from .config import ChainLiteConfig


def load_yaml_config_data(yaml_file_path: str) -> dict[str, Any]:
    """Load raw config dictionary from a YAML file."""
    with open(yaml_file_path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML config must decode to a mapping/dict, got {type(data).__name__}"
        )
    return data


def normalize_config_data(
    config_data: dict[str, Any],
    source_path: str,
    custom_configs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply runtime overrides and defaults to config data."""
    normalized = dict(config_data)
    if custom_configs is not None:
        normalized.update(custom_configs)

    if normalized.get("config_name") is None:
        normalized["config_name"] = str(Path(source_path))
    return normalized


def load_chainlite_from_yaml(
    yaml_file_path: str, custom_configs: Optional[dict[str, Any]] = None
):
    """Build a ChainLite instance from a YAML config file."""
    yaml_data = load_yaml_config_data(yaml_file_path)
    normalized_data = normalize_config_data(yaml_data, yaml_file_path, custom_configs)

    # Local import avoids runtime circular import with core module.
    from .core import ChainLite

    return ChainLite(ChainLiteConfig(**normalized_data))
