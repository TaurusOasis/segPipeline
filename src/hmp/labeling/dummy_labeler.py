"""DummyLabeler (Step 06).

Creates a centered rectangular person mask per image. Used for tests and the
CPU-only demo pipeline so the downstream stages (refine, export, trimap, report)
can run without any model weights or GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..data.mask_io import mask_to_bbox_xyxy, write_binary_mask
from ..schemas import InstanceAnnotation, MediaItem
from .base import Labeler


class DummyLabeler(Labeler):
    """Centered-rectangle person mask labeler."""

    name = "dummy"

    def __init__(self, cfg, *, project_root: Optional[Path] = None, **kw) -> None:
        super().__init__(cfg, project_root=project_root, **kw)
        dcfg = cfg.get("dummy", {}) if hasattr(cfg, "get") else {}
        # fraction of image the central rectangle occupies (in each axis)
        self.fx = float(dcfg.get("width_fraction", 0.4))
        self.fy = float(dcfg.get("height_fraction", 0.6))
        self.score = float(dcfg.get("score", 0.95))

    def label_one(self, item: MediaItem) -> list[InstanceAnnotation]:
        h, w = item.height, item.width
        bh = max(1, int(round(h * self.fy)))
        bw = max(1, int(round(w * self.fx)))
        y0 = (h - bh) // 2
        x0 = (w - bw) // 2
        mask = np.zeros((h, w), bool)
        mask[y0 : y0 + bh, x0 : x0 + bw] = True

        mask_path = self.mask_dir / f"{item.item_id}_person_0.png"
        write_binary_mask(mask_path, mask)
        bbox = mask_to_bbox_xyxy(mask) or [x0, y0, x0 + bw, y0 + bh]
        return [
            InstanceAnnotation(
                instance_id="person_0",
                category="person",
                bbox_xyxy=bbox,
                mask_path=str(mask_path),
                score=self.score,
                source="dummy",
            )
        ]