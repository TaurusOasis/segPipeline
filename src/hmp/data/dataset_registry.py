"""Load the machine-readable dataset registry from configs/datasets.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import Config


def default_registry_path(project_root: Path) -> Path:
    return project_root / "configs" / "datasets.yaml"


def load_dataset_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset registry not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"dataset registry root must be a mapping: {path}")
    return data


def get_dataset_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = registry.get("datasets", {})
    if dataset_id not in datasets:
        known = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"unknown dataset_id={dataset_id!r}; known: {known}")
    entry = datasets[dataset_id]
    if not isinstance(entry, dict):
        raise ValueError(f"dataset entry must be a mapping: {dataset_id}")
    return entry


def registry_path_from_config(cfg: Config, project_root: Path) -> Path:
    ingest = cfg.get("ingest", {})
    explicit = ingest.get("registry_path")
    if explicit:
        return Path(explicit) if Path(explicit).is_absolute() else project_root / explicit
    return default_registry_path(project_root)
