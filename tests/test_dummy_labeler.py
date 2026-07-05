"""Tests for hmp.labeling (Step 06)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config
from hmp.data.build_manifest import build_media_items
from hmp.labeling.dummy_labeler import DummyLabeler
from hmp.labeling.export_annotations import filter_with_instances, summary
from hmp.schemas import AnnotationRecord, InstanceAnnotation, MediaItem


def _make_images(tmp_path, n=3, w=40, h=60):
    raw = tmp_path / "raw"
    files = []
    for i in range(n):
        p = raw / f"img{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), (10 * i, 20 * i, 30 * i)).save(p)
        files.append(p)
    return raw, files


def test_dummy_labeler_produces_masks_and_annotations(tmp_path):
    raw, files = _make_images(tmp_path)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)

    mask_dir = tmp_path / "masks"
    ann = tmp_path / "ann.jsonl"
    cfg = Config({"paths": {"masks_raw_dir": str(mask_dir), "annotation_path": str(ann)}})
    labeler = DummyLabeler(cfg, project_root=tmp_path)
    out = labeler.run(manifest, overwrite=True)
    assert out == ann

    recs = read_jsonl_list(ann, model=AnnotationRecord)
    assert len(recs) == 3
    for r in recs:
        assert len(r.instances) == 1
        inst = r.instances[0]
        assert inst.category == "person"
        assert inst.source == "dummy"
        assert Path(inst.mask_path).exists()
        assert inst.bbox_xyxy[2] > inst.bbox_xyxy[0]
        assert inst.bbox_xyxy[3] > inst.bbox_xyxy[1]


def test_dummy_labeler_central_rectangle(tmp_path):
    raw, files = _make_images(tmp_path, n=1, w=40, h=60)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)
    mask_dir = tmp_path / "masks"
    ann = tmp_path / "ann.jsonl"
    cfg = Config({"paths": {"masks_raw_dir": str(mask_dir), "annotation_path": str(ann)}})
    DummyLabeler(cfg, project_root=tmp_path).run(manifest, overwrite=True)
    recs = read_jsonl_list(ann, model=AnnotationRecord)
    mask = recs[0].instances[0]
    # width_fraction=0.4 -> bw=16, height_fraction=0.6 -> bh=36
    bx1, by1, bx2, by2 = mask.bbox_xyxy
    assert (bx2 - bx1) == 16
    assert (by2 - by1) == 36


def test_dummy_labeler_dry_run(tmp_path):
    raw, files = _make_images(tmp_path, n=2)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)
    cfg = Config({"paths": {"masks_raw_dir": str(tmp_path / "masks"), "annotation_path": str(tmp_path / "ann.jsonl")}})
    labeler = DummyLabeler(cfg, project_root=tmp_path)
    labeler.run(manifest, dry_run=True)
    assert not (tmp_path / "ann.jsonl").exists()


def test_summary_and_filter(tmp_path):
    ann = tmp_path / "ann.jsonl"
    recs = [
        AnnotationRecord(item_id="a", instances=[InstanceAnnotation(instance_id="p0", bbox_xyxy=[1, 1, 2, 2], source="dummy")]),
        AnnotationRecord(item_id="b", instances=[]),
        AnnotationRecord(item_id="c", instances=[InstanceAnnotation(instance_id="p0", bbox_xyxy=[1, 1, 2, 2], source="sam3")]),
    ]
    write_jsonl(ann, recs, overwrite=True)
    s = summary(ann)
    assert s["n_items"] == 3
    assert s["n_items_with_instances"] == 2
    assert s["n_instances"] == 2
    assert s["by_source"] == {"dummy": 1, "sam3": 1}

    out = tmp_path / "filtered.jsonl"
    n = filter_with_instances(ann, out)
    assert n == 2
    kept = read_jsonl_list(out, model=AnnotationRecord)
    assert {r.item_id for r in kept} == {"a", "c"}