"""Video I/O helpers (Step 03).

Thin, dependency-light wrapper around OpenCV (cv2). cv2 is a lightweight
runtime dep already in requirements/base.txt — it is NOT a heavy GPU dep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


def _lazy_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "opencv-python is required for video I/O: pip install -r requirements/base.txt"
        ) from e
    return cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def iter_video_files(root: str | Path, extensions: Optional[set[str]] = None) -> list[Path]:
    exts = {e.lower() for e in (extensions or VIDEO_EXTENSIONS)}
    root = Path(root)
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def open_video(path: str | Path):
    """Open a video capture; raises if the file cannot be opened."""
    cv2 = _lazy_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def video_metadata(path: str | Path) -> dict:
    """Return fps, frame_count, width, height for a video."""
    cap = open_video(path)
    try:
        fps = cap.get(5) or 0.0  # CAP_PROP_FPS
        n = int(cap.get(7) or 0)  # CAP_PROP_FRAME_COUNT
        w = int(cap.get(3) or 0)  # CAP_PROP_FRAME_WIDTH
        h = int(cap.get(4) or 0)  # CAP_PROP_FRAME_HEIGHT
    finally:
        cap.release()
    return {"fps": float(fps), "frame_count": n, "width": w, "height": h}


def iter_frame_indices(
    frame_count: int,
    fps: float,
    *,
    every_n_frames: Optional[int] = None,
    fps_target: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> list[int]:
    """Decide which frame indices to extract (deterministic)."""
    if frame_count <= 0:
        return []
    if every_n_frames is not None and every_n_frames > 0:
        step = int(every_n_frames)
    elif fps_target is not None and fps_target > 0 and fps > 0:
        step = max(1, int(round(fps / fps_target)))
    else:
        step = 1
    idxs = list(range(0, frame_count, step))
    if max_frames is not None and max_frames > 0:
        idxs = idxs[:max_frames]
    return idxs