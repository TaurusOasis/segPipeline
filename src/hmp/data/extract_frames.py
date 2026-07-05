"""Extract video frames and append MediaItem records to the manifest (Step 03)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..common.hashing import sha256_file
from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..common.video_io import iter_frame_indices, iter_video_files, open_video
from ..config import Config, resolve_path
from ..schemas import MediaItem

log = get_logger("hmp.data.extract_frames")


def extract_video(
    video_path: Path,
    out_dir: Path,
    *,
    every_n_frames: Optional[int] = None,
    fps_target: Optional[float] = None,
    max_frames: Optional[int] = None,
    dry_run: bool = False,
) -> list[MediaItem]:
    """Extract frames from one video into ``out_dir``.

    Frames are written as ``frame_000001.jpg``. Returns MediaItem records
    (media_type="frame") with source_video, frame_index, timestamp_ms set.
    """
    cv2 = __import__("cv2")  # local; also lazy-imported via open_video
    cap = open_video(video_path)
    try:
        fps = float(cap.get(5) or 0.0) or 25.0
        frame_count = int(cap.get(7) or 0)
        idxs = iter_frame_indices(
            frame_count, fps,
            every_n_frames=every_n_frames,
            fps_target=fps_target,
            max_frames=max_frames,
        )
        items: list[MediaItem] = []
        if dry_run:
            log.info("[dry-run] would extract %d frames from %s", len(idxs), video_path)
            return items

        out_dir.mkdir(parents=True, exist_ok=True)
        video_stem = video_path.stem
        # Seek frame-by-frame to desired indices for determinism.
        desired = set(idxs)
        next_i = 0
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in desired:
                fname = out_dir / f"frame_{next_i + 1:06d}.jpg"
                cv2.imwrite(str(fname), frame)
                w = frame.shape[1]
                h = frame.shape[0]
                ts_ms = int(round(fi / fps * 1000)) if fps > 0 else 0
                items.append(
                    MediaItem(
                        item_id=f"{video_stem}_f{fi:06d}",
                        media_type="frame",
                        path=str(fname),
                        width=w,
                        height=h,
                        sha256=sha256_file(fname),
                        source_video=str(video_path),
                        frame_index=fi,
                        timestamp_ms=ts_ms,
                        tags=["video_frame"],
                    )
                )
                next_i += 1
            fi += 1
    finally:
        cap.release()
    log.info("Extracted %d frames from %s -> %s", len(items), video_path, out_dir)
    return items


def extract_frames(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    """Extract frames from all videos under raw_dir into frames_dir."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    raw_dir = resolve_path(root, paths.get("raw_dir", "data/raw"))
    frames_dir = resolve_path(root, paths.get("frames_dir", "data/frames"))
    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))

    fx_cfg = cfg.get("frames", {})
    every_n = fx_cfg.get("every_n_frames")
    fps_target = fx_cfg.get("fps")
    max_frames = fx_cfg.get("max_frames_per_video")

    videos = iter_video_files(raw_dir)
    log.info("Found %d videos under %s", len(videos), raw_dir)

    all_items: list[MediaItem] = []
    if not dry_run and manifest_path.exists() and not overwrite:
        # append mode: keep existing items, only add new frames
        existing = read_jsonl_list(manifest_path, model=MediaItem)
        all_items.extend(existing)

    for v in videos:
        out_dir = frames_dir / v.stem
        items = extract_video(
            v, out_dir,
            every_n_frames=every_n,
            fps_target=fps_target,
            max_frames=max_frames,
            dry_run=dry_run,
        )
        all_items.extend(items)

    if dry_run:
        return manifest_path

    write_jsonl(manifest_path, all_items, overwrite=True)
    log.info("Manifest now has %d items at %s", len(all_items), manifest_path)
    return manifest_path