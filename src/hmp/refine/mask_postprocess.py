"""Mask postprocessing operations (Step 04).

Thin, pure-numpy/cv2 wrappers around the helpers in :mod:`hmp.data.mask_io`,
exposed as a single config-driven entry point used by the refine stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.mask_io import fill_holes, keep_largest_component, remove_small_components


def postprocess_mask(
    mask: np.ndarray,
    *,
    remove_small: bool = True,
    min_component_area: int = 64,
    fill: bool = True,
    keep_largest: bool = False,
) -> np.ndarray:
    """Apply a fixed order of local postprocess operations to a binary mask."""
    out = np.asarray(mask) > 0
    if remove_small:
        out = remove_small_components(out, min_area=min_component_area)
    if fill:
        out = fill_holes(out)
    if keep_largest:
        out = keep_largest_component(out)
    return out


def postprocess_from_config(mask: np.ndarray, cfg: Any) -> np.ndarray:
    """Read a ``local_postprocess`` config block and run :func:`postprocess_mask`."""
    lp = cfg.get("local_postprocess", {}) if hasattr(cfg, "get") else {}
    return postprocess_mask(
        mask,
        remove_small=bool(lp.get("remove_small_components", True)),
        min_component_area=int(lp.get("min_component_area", 64)),
        fill=bool(lp.get("fill_holes", True)),
        keep_largest=bool(lp.get("keep_largest_component", False)),
    )