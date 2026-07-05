"""Tests for YOLO seg export (Step 07)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hmp.config import Config
from hmp.data.yolo_seg_io import (
    mask_to_polygons,
    normalize_polygon,
    polygon_to_yolo_line,
    read_yolo_label,
    write_data_yaml,
    write_yolo_label,
)
from hmp.data.split_dataset import train_val_split
from hmp.labeling.dummy_labeler import DummyLabeler
from hmp.data.build_manifest import build_media_items
from hmp.common.jsonl import write_jsonl
from hmp.yolo.export_yolo_dataset import export_yolo_dataset


def _make_dataset(tmp_path, n=5, w=40, h=60):
    raw = tmp_path / "raw"
    files = []
    for i in range(n):
        p = raw / f"img{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), (i * 10, 0, 0)).save(p)
        files.append(p)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)
    cfg = Config({"paths": {"masks_raw_dir": str(tmp_path / "masks"), "annotation_path": str(tmp_path / "ann.jsonl")}})
    DummyLabeler(cfg, project_root=tmp_path).run(manifest, overwrite=True)
    return items, manifest, tmp_path / "ann.jsonl"


def test_mask_to_polygons_and_normalize():
    import numpy as np

    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    polys = mask_to_polygons(m)
    assert len(polys) >= 1
    norm = normalize_polygon(polys[0], 40, 40)
    assert norm.max() <= 1.0 and norm.min() >= 0.0


def test_polygon_to_yolo_line():
    import numpy as np

    poly = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    line = polygon_to_yolo_line(0, poly)
    assert line.startswith("0 ")
    assert "0.000000" in line


def test_write_and_read_yolo_label(tmp_path):
    import numpy as np

    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    p = tmp_path / "label.txt"
    n = write_yolo_label(p, 0, m, 40, 40)
    assert n >= 1
    parsed = read_yolo_label(p)
    assert len(parsed) == n
    assert parsed[0][0] == 0
    assert parsed[0][1].shape[1] == 2


def test_read_yolo_label_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    assert read_yolo_label(p) == []


def test_train_val_split_deterministic():
    items = list(range(100))
    tr1, va1 = train_val_split(items, val_ratio=0.2, seed=42)
    tr2, va2 = train_val_split(items, val_ratio=0.2, seed=42)
    assert tr1 == tr2 and va1 == va2
    assert len(va1) == 20 and len(tr1) == 80
    assert set(tr1) | set(va1) == set(items)
    assert not (set(tr1) & set(va1))


def test_export_yolo_dataset_structure(tmp_path):
    items, manifest, ann = _make_dataset(tmp_path, n=5)
    yolo_dir = tmp_path / "yolo"
    cfg = Config(
        {
            "paths": {
                "manifest_path": str(manifest),
                "annotation_path": str(ann),
                "yolo_dir": str(yolo_dir),
            },
            "yolo_export": {"val_ratio": 0.2, "seed": 42, "symlink": False, "class_names": ["person"]},
        }
    )
    out = export_yolo_dataset(cfg, project_root=tmp_path)
    assert out == yolo_dir
    assert (yolo_dir / "data.yaml").exists()
    assert (yolo_dir / "images" / "train").is_dir()
    assert (yolo_dir / "labels" / "train").is_dir()
    # each image has a label file
    imgs = list((yolo_dir / "images" / "train").iterdir()) + list((yolo_dir / "images" / "val").iterdir())
    labels = list((yolo_dir / "labels" / "train").iterdir()) + list((yolo_dir / "labels" / "val").iterdir())
    assert len(imgs) == 5
    assert len(labels) == 5


def test_export_yolo_dataset_valid_labels(tmp_path):
    items, manifest, ann = _make_dataset(tmp_path, n=3)
    yolo_dir = tmp_path / "yolo"
    cfg = Config(
        {
            "paths": {"manifest_path": str(manifest), "annotation_path": str(ann), "yolo_dir": str(yolo_dir)},
            "yolo_export": {"val_ratio": 0.34, "seed": 1, "symlink": False, "class_names": ["person"]},
        }
    )
    export_yolo_dataset(cfg, project_root=tmp_path)
    # at least one non-empty label exists
    non_empty = []
    for split in ("train", "val"):
        for lf in (yolo_dir / "labels" / split).iterdir():
            parsed = read_yolo_label(lf)
            if parsed:
                non_empty.append(parsed)
    assert non_empty
    for parsed in non_empty:
        cls, poly = parsed[0]
        assert cls == 0
        assert poly.min() >= 0.0 and poly.max() <= 1.0


def test_export_yolo_dataset_dry_run(tmp_path):
    items, manifest, ann = _make_dataset(tmp_path, n=2)
    yolo_dir = tmp_path / "yolo"
    cfg = Config({"paths": {"manifest_path": str(manifest), "annotation_path": str(ann), "yolo_dir": str(yolo_dir)}})
    out = export_yolo_dataset(cfg, project_root=tmp_path, dry_run=True)
    assert out == yolo_dir
    assert not yolo_dir.exists() or not any(yolo_dir.iterdir())


def test_empty_annotations_produce_empty_label(tmp_path):
    # build manifest with images but an annotation file that's empty
    raw = tmp_path / "raw"
    files = []
    for i in range(2):
        p = raw / f"img{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), (0, 0, 0)).save(p)
        files.append(p)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)
    ann = tmp_path / "ann.jsonl"
    ann.write_text("", encoding="utf-8")  # empty annotations
    yolo_dir = tmp_path / "yolo"
    cfg = Config(
        {
            "paths": {"manifest_path": str(manifest), "annotation_path": str(ann), "yolo_dir": str(yolo_dir)},
            "yolo_export": {"val_ratio": 0.5, "seed": 1, "symlink": False},
        }
    )
    export_yolo_dataset(cfg, project_root=tmp_path)
    for split in ("train", "val"):
        for lf in (yolo_dir / "labels" / split).iterdir():
            assert lf.stat().st_size == 0  # empty label