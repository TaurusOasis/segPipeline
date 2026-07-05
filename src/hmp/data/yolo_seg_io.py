"""YOLO segmentation label IO (Step 07).

YOLO seg label line format::

    <class_id> x1 y1 x2 y2 x3 y3 ... xn yn

All polygon coordinates normalized to [0, 1]. Masks -> polygons via OpenCV
contours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def mask_to_polygons(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Convert a binary mask to a list of normalized polygons.

    Each polygon is an Nx2 float array. Coordinates are absolute pixels here;
    normalization happens in :func:`normalize_polygon`.
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.reshape(-1, 2)
        if c.shape[0] < 3:
            continue
        if cv2.contourArea(c.astype(np.int32)) >= min_area:
            polys.append(c)
    return polys


def normalize_polygon(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize absolute pixel polygon coords to [0, 1] using width/height."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    out = p.copy()
    out[:, 0] /= float(width)
    out[:, 1] /= float(height)
    return np.clip(out, 0.0, 1.0)


def polygon_to_yolo_line(class_id: int, poly_norm: np.ndarray) -> str:
    """Format a normalized polygon as a YOLO seg label line."""
    coords = poly_norm.reshape(-1)
    coords_str = " ".join(f"{v:.6f}" for v in coords)
    return f"{int(class_id)} {coords_str}"


def write_yolo_label(path: str | Path, class_id: int, mask: np.ndarray, width: int, height: int) -> int:
    """Write a YOLO seg label file for one image's single-class mask.

    Returns the number of polygons written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    polys = mask_to_polygons(mask)
    lines = []
    for p in polys:
        norm = normalize_polygon(p, width, height)
        lines.append(polygon_to_yolo_line(class_id, norm))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    return len(lines)


def read_yolo_label(path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Parse a YOLO seg label file into (class_id, polygon) pairs."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            out.append((cls, coords))
    return out


def write_data_yaml(path: str | Path, *, yolo_dir: Path, class_names: list[str], train: str = "images/train", val: str = "images/val") -> None:
    """Write the Ultralytics ``data.yaml`` for a YOLO seg dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {Path(yolo_dir).resolve()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"names:\n"
    )
    for i, name in enumerate(class_names):
        content += f"  {i}: {name}\n"
    path.write_text(content, encoding="utf-8")