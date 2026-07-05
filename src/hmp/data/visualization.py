"""Dataset visualization helpers (Step 07)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def draw_mask_on_image(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on an image (BGR or RGB)."""
    img = image.copy()
    m = np.asarray(mask) > 0
    color_arr = np.array(color, dtype=image.dtype)
    overlay = img.copy()
    overlay[m] = color_arr
    out = np.clip(image * (1 - alpha) + overlay * alpha, 0, image.max() if image.dtype == np.uint8 else 1.0).astype(image.dtype)
    return out


def save_visualization(image_path: Path, mask: np.ndarray, out_path: Path) -> None:
    """Save a side-by-side image+mask overlay PNG."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    vis = draw_mask_on_image(img, mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)