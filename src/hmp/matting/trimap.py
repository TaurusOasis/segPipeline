"""Trimap generation from binary masks (Step 17).

Trimap values: 0 = background, 128 = unknown (boundary band), 255 = foreground.
The unknown band is the region between an eroded foreground and a dilated
foreground, giving a configurable boundary band around the mask contour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import read_binary_mask, write_uint8_image
from ..schemas import AnnotationRecord

log = get_logger("hmp.matting.trimap")


def make_trimap(mask: np.ndarray, radius: int = 12) -> np.ndarray:
    """Build a uint8 trimap from a boolean mask.

    ``radius`` controls the unknown-band width (in pixels, applied symmetrically:
    erode foreground by ``radius`` to get sure-fg, dilate by ``radius`` to get
    the outer boundary of unknown).
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    r = max(1, int(radius))
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.erode(m, kernel, iterations=r)   # sure foreground (interior)
    dil = cv2.dilate(m, kernel, iterations=r)   # enlarged foreground
    trimap = np.zeros_like(m, dtype=np.uint8)        # background
    trimap[dil > 0] = 128                            # unknown band (dilated minus eroded)
    trimap[fg > 0] = 255                             # sure foreground
    return trimap


def make_trimap_from_annotation(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    """Generate trimaps for all masks referenced in the annotation JSONL."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    annotation_path = resolve_path(
        root, paths.get("refined_annotation_path", paths.get("annotation_path", "data/annotations/annotations_refined.jsonl"))
    )
    tcfg = cfg.get("trimap", {})
    out_dir = resolve_path(root, tcfg.get("output_dir", "data/alpha/trimaps"))
    radius = int(tcfg.get("radius", 12))

    records = read_jsonl_list(annotation_path, model=AnnotationRecord)
    log.info("Generating trimaps for %d items (radius=%d)", len(records), radius)

    if dry_run:
        log.info("[dry-run] would write trimaps to %s", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in records:
        for inst in rec.instances:
            if not inst.mask_path:
                continue
            mask = read_binary_mask(inst.mask_path)
            tri = make_trimap(mask, radius=radius)
            out_path = out_dir / f"{rec.item_id}_{inst.instance_id}_trimap.png"
            write_uint8_image(out_path, tri)  # preserves 0/128/255 values
            n += 1
    log.info("Wrote %d trimaps to %s", n, out_dir)
    return out_dir