"""Tests for hmp.data.extract_frames (Step 03)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hmp.common.jsonl import read_jsonl_list
from hmp.common.video_io import iter_frame_indices
from hmp.config import Config
from hmp.data.extract_frames import extract_frames, extract_video
from hmp.schemas import MediaItem


def _make_video(path: Path, n_frames: int = 10, w: int = 16, h: int = 12, fps: float = 25.0):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_iter_frame_indices_every_n():
    assert iter_frame_indices(10, 25.0, every_n_frames=2) == [0, 2, 4, 6, 8]


def test_iter_frame_indices_fps_target():
    # 25 fps, target 5 fps -> step 5
    assert iter_frame_indices(25, 25.0, fps_target=5.0) == [0, 5, 10, 15, 20]


def test_iter_frame_indices_max_frames():
    idxs = iter_frame_indices(100, 25.0, every_n_frames=1, max_frames=3)
    assert idxs == [0, 1, 2]


def test_iter_frame_indices_empty():
    assert iter_frame_indices(0, 25.0) == []


def test_extract_video_writes_frames(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, n_frames=6, w=20, h=14)
    out = tmp_path / "frames" / "v"
    items = extract_video(v, out, every_n_frames=2)
    assert len(items) == 3  # frames 0,2,4
    assert all(isinstance(i, MediaItem) for i in items)
    assert all(i.media_type == "frame" for i in items)
    assert all(Path(i.path).exists() for i in items)
    assert items[0].frame_index == 0
    assert items[-1].frame_index == 4
    assert items[0].source_video == str(v)
    assert items[0].width == 20 and items[0].height == 14


def test_extract_video_dry_run(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, n_frames=4, w=8, h=8)
    out = tmp_path / "frames" / "v"
    items = extract_video(v, out, every_n_frames=1, dry_run=True)
    assert items == []
    assert not out.exists() or not any(out.iterdir())


def test_extract_frames_appends_manifest(tmp_path):
    raw = tmp_path / "raw"
    v = raw / "clip.mp4"
    _make_video(v, n_frames=4, w=8, h=8)
    manifest = tmp_path / "m.jsonl"
    cfg = Config(
        {
            "paths": {
                "raw_dir": str(raw),
                "frames_dir": str(tmp_path / "frames"),
                "manifest_path": str(manifest),
            },
            "frames": {"every_n_frames": 1},
        }
    )
    out = extract_frames(cfg, project_root=tmp_path, overwrite=True)
    assert out == manifest
    items = read_jsonl_list(manifest, model=MediaItem)
    assert len(items) == 4
    assert all(i.media_type == "frame" for i in items)