"""Image I/O helpers (Step 02).

Uses Pillow for reading dimensions and format, which keeps this lightweight
(no torch / cv2 required for manifest building). OpenCV is used elsewhere for
mask/contour work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_image_size(path: str | Path) -> tuple[int, int]:
    """Return (width, height) of an image without fully decoding it."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def read_image(path: str | Path) -> "Image.Image":
    """Load an image as a PIL RGB image."""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def iter_image_files(
    root: str | Path,
    extensions: Optional[set[str]] = None,
) -> list[Path]:
    """Recursively list image files under ``root`` sorted by path."""
    exts = {e.lower() for e in (extensions or IMAGE_EXTENSIONS)}
    root = Path(root)
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files