"""Adaptive matting-critical region generation (pipeline step 6).

Replaces fixed erode/dilate with distance-transform-based unknown bands that
can widen near boundary pixels. Optional motion/hair flags widen the band
further without requiring GPU models in the default CPU path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.jsonl import read_jsonl_list
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import read_binary_mask, write_uint8_image
from ..schemas import AnnotationRecord
from .trimap import make_trimap

log = get_logger("hmp.matting.adaptive_trimap")

TRIMapValues = {"background": 0, "unknown": 128, "foreground": 255}


def make_adaptive_trimap(
    mask: np.ndarray,
    *,
    base_radius: int = 12,
    max_radius: int = 24,
    hair_priority: bool = False,
    motion_blur: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build trimap + ROI cores from a boolean mask.

    Returns:
        trimap: uint8 array with values 0 / 128 / 255
        roi: dict with foreground_core, background_core, unknown_roi (bool arrays)
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        empty = np.zeros_like(m, dtype=np.uint8)
        roi = {
            "foreground_core": empty.astype(bool),
            "background_core": ~empty.astype(bool),
            "unknown_roi": empty.astype(bool),
        }
        return empty, roi

    radius_boost = 0
    if hair_priority:
        radius_boost += 4
    if motion_blur:
        radius_boost += 6

    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    boundary_dist = np.minimum(dist_in, dist_out)
    local_radius = np.clip(base_radius + radius_boost + (4 - np.minimum(boundary_dist, 4)), 1, max_radius)

    fg_core = dist_in >= local_radius
    bg_core = dist_out >= local_radius
    unknown = ~(fg_core | bg_core)

    trimap = np.zeros_like(m, dtype=np.uint8)
    trimap[unknown] = TRIMapValues["unknown"]
    trimap[fg_core] = TRIMapValues["foreground"]
    trimap[bg_core] = TRIMapValues["background"]

    roi = {
        "foreground_core": fg_core.astype(bool),
        "background_core": bg_core.astype(bool),
        "unknown_roi": unknown.astype(bool),
    }
    return trimap, roi


def _hair_or_motion_from_tags(tags: list[str]) -> tuple[bool, bool]:
    hair = any("hair" in t.lower() for t in tags)
    motion = any("motion" in t.lower() or "blur" in t.lower() for t in tags)
    return hair, motion


def make_adaptive_trimap_from_annotation(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    """Generate adaptive trimaps and ROI sidecars for annotated instances."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    tcfg = cfg.get("adaptive_trimap", cfg.get("trimap", {}))
    annotation_path = resolve_path(
        root,
        paths.get("refined_annotation_path", paths.get("annotation_path", "data/annotations/annotations_refined.jsonl")),
    )
    out_dir = resolve_path(root, tcfg.get("output_dir", "data/alpha/adaptive_trimaps"))
    roi_dir = resolve_path(root, tcfg.get("roi_dir", "data/alpha/roi"))
    base_radius = int(tcfg.get("base_radius", tcfg.get("radius", 12)))
    max_radius = int(tcfg.get("max_radius", 24))

    records = read_jsonl_list(annotation_path, model=AnnotationRecord)
    log.info("Generating adaptive trimaps for %d items", len(records))

    if dry_run:
        log.info("[dry-run] would write adaptive trimaps to %s and ROI to %s", out_dir, roi_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in records:
        for inst in rec.instances:
            if not inst.mask_path:
                continue
            mask = read_binary_mask(inst.mask_path)
            hair, motion = _hair_or_motion_from_tags(getattr(inst, "tags", []) or [])
            tri, roi = make_adaptive_trimap(
                mask,
                base_radius=base_radius,
                max_radius=max_radius,
                hair_priority=hair,
                motion_blur=motion,
            )
            stem = f"{rec.item_id}_{inst.instance_id}"
            write_uint8_image(out_dir / f"{stem}_trimap.png", tri)
            write_uint8_image(roi_dir / f"{stem}_fg_core.png", roi["foreground_core"].astype(np.uint8) * 255)
            write_uint8_image(roi_dir / f"{stem}_bg_core.png", roi["background_core"].astype(np.uint8) * 255)
            write_uint8_image(roi_dir / f"{stem}_unknown_roi.png", roi["unknown_roi"].astype(np.uint8) * 255)
            n += 1
    log.info("Wrote %d adaptive trimaps to %s", n, out_dir)
    return out_dir


def fallback_fixed_trimap(mask: np.ndarray, radius: int = 12) -> np.ndarray:
    """Compatibility wrapper around the legacy fixed-band trimap generator."""
    return make_trimap(mask, radius=radius)
