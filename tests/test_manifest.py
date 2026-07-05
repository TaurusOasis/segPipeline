"""Tests for hmp.data.build_manifest (Step 02)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hmp.common.jsonl import read_jsonl_list
from hmp.config import Config
from hmp.data.build_manifest import build_manifest, build_media_items, item_id_from_path
from hmp.schemas import MediaItem


def _make_image(path: Path, w: int = 32, h: int = 24, color=(128, 64, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def test_item_id_from_path(tmp_path):
    root = tmp_path
    p = tmp_path / "sub" / "img.jpg"
    p.parent.mkdir(parents=True)
    assert item_id_from_path(root, p) == "sub_img"


def test_build_media_items(tmp_path):
    root = tmp_path / "raw"
    f1 = root / "a.jpg"
    f2 = root / "b.png"
    _make_image(f1, 30, 20)
    _make_image(f2, 64, 48)
    items = build_media_items(root, [f1, f2], tags=["raw"])
    assert len(items) == 2
    assert items[0].width == 30 and items[0].height == 20
    assert items[1].width == 64
    assert all(isinstance(i, MediaItem) for i in items)
    assert all(len(i.sha256) == 64 for i in items)
    assert items[0].tags == ["raw"]


def test_build_manifest_writes_jsonl(tmp_path):
    raw = tmp_path / "raw"
    _make_image(raw / "a.jpg", 10, 10)
    _make_image(raw / "deep" / "b.webp", 12, 12)
    manifest = tmp_path / "manifests" / "m.jsonl"

    cfg = Config(
        {
            "paths": {
                "raw_dir": str(raw),
                "manifest_path": str(manifest),
            }
        }
    )
    out = build_manifest(cfg, project_root=tmp_path, overwrite=True)
    assert out == manifest
    items = read_jsonl_list(manifest, model=MediaItem)
    assert {i.item_id for i in items} == {"a", "deep_b"}
    assert all(i.media_type == "image" for i in items)
    assert all(i.width >= 1 and i.height >= 1 for i in items)


def test_build_manifest_no_overwrite_keeps_existing(tmp_path):
    raw = tmp_path / "raw"
    _make_image(raw / "a.jpg", 8, 8)
    manifest = tmp_path / "m.jsonl"
    cfg = Config({"paths": {"raw_dir": str(raw), "manifest_path": str(manifest)}})

    build_manifest(cfg, project_root=tmp_path, overwrite=True)
    first = read_jsonl_list(manifest, model=MediaItem)
    # add a second image and rebuild without overwrite -> existing kept
    _make_image(raw / "b.jpg", 8, 8)
    build_manifest(cfg, project_root=tmp_path, overwrite=False)
    second = read_jsonl_list(manifest, model=MediaItem)
    assert first == second  # unchanged
    assert len(second) == 1


def test_build_manifest_dry_run_no_file(tmp_path):
    raw = tmp_path / "raw"
    _make_image(raw / "a.jpg", 8, 8)
    manifest = tmp_path / "m.jsonl"
    cfg = Config({"paths": {"raw_dir": str(raw), "manifest_path": str(manifest)}})
    out = build_manifest(cfg, project_root=tmp_path, dry_run=True)
    assert not manifest.exists()
    assert out == manifest