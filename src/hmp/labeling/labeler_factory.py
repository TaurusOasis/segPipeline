"""Select labeler implementation for pipeline step 2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import Config
from .base import Labeler
from .dummy_labeler import DummyLabeler
from .yolo_sam2_labeler import YoloSam2Labeler

ProviderName = Literal["mock", "yolo_sam2", "yolo_grabcut"]


def make_labeler(cfg: Config, *, project_root: Path, provider: str = "mock") -> Labeler:
    if provider == "mock":
        return DummyLabeler(cfg, project_root=project_root)
    if provider == "yolo_grabcut":
        return YoloSam2Labeler(cfg, project_root=project_root, segment_mode="grabcut")
    if provider == "yolo_sam2":
        return YoloSam2Labeler(cfg, project_root=project_root, segment_mode="sam2")
    raise ValueError(f"unknown labeling provider: {provider!r}; use mock | yolo_grabcut | yolo_sam2")
