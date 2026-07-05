"""Tests for local mask refinement pipeline (Step 12 minimal)."""

from __future__ import annotations

import numpy as np

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config
from hmp.data.build_manifest import build_media_items
from hmp.data.mask_io import write_binary_mask
from hmp.labeling.dummy_labeler import DummyLabeler
from hmp.refine.refine_pipeline import refine_masks
from hmp.schemas import AnnotationRecord, InstanceAnnotation


def _setup(tmp_path, n=4, w=40, h=50):
    from PIL import Image

    raw = tmp_path / "raw"
    files = []
    for i in range(n):
        p = raw / f"img{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), (i, i, i)).save(p)
        files.append(p)
    items = build_media_items(raw, files)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, items, overwrite=True)
    cfg = Config(
        {
            "paths": {
                "masks_raw_dir": str(tmp_path / "masks_raw"),
                "masks_refined_dir": str(tmp_path / "masks_refined"),
                "annotation_path": str(tmp_path / "ann_raw.jsonl"),
                "refined_annotation_path": str(tmp_path / "ann_refined.jsonl"),
            },
            "local_postprocess": {"remove_small_components": True, "min_component_area": 16, "fill_holes": True, "keep_largest_component": False},
            "refine": {"report_path": str(tmp_path / "refine_report.jsonl")},
        }
    )
    DummyLabeler(cfg, project_root=tmp_path).run(manifest, overwrite=True)
    return cfg


def test_refine_writes_refined_annotations_and_report(tmp_path):
    cfg = _setup(tmp_path, n=4)
    ann, report = refine_masks(cfg, project_root=tmp_path)
    assert ann.exists() and report.exists()
    recs = read_jsonl_list(ann, model=AnnotationRecord)
    assert len(recs) == 4
    for r in recs:
        for inst in r.instances:
            assert inst.mask_path is not None
            assert "masks_refined" in inst.mask_path
            assert (inst.source or "").endswith("+refined_local")
    reports = read_jsonl_list(report)
    assert len(reports) == 4
    assert all("area_before" in x and "area_after" in x for x in reports)


def test_refine_removes_small_components(tmp_path):
    cfg = _setup(tmp_path, n=1)
    # add a tiny speckle to the raw mask of item 0
    recs = read_jsonl_list(tmp_path / "ann_raw.jsonl", model=AnnotationRecord)
    inst = recs[0].instances[0]
    m = np.zeros((50, 40), bool)
    m[10:30, 10:30] = True
    m[1:2, 1:2] = True  # tiny speck (1 px) < min_component_area 16
    write_binary_mask(inst.mask_path, m)
    refine_masks(cfg, project_root=tmp_path)
    refined = read_jsonl_list(tmp_path / "ann_refined.jsonl", model=AnnotationRecord)[0].instances[0]
    from hmp.data.mask_io import read_binary_mask

    out = read_binary_mask(refined.mask_path)
    assert not out[1, 2]  # speck removed


def test_refine_dry_run(tmp_path):
    cfg = _setup(tmp_path, n=2)
    ann, report = refine_masks(cfg, project_root=tmp_path, dry_run=True)
    assert not (tmp_path / "ann_refined.jsonl").exists()


def test_refine_empty_instances_passthrough(tmp_path):
    ann_raw = tmp_path / "ann_raw.jsonl"
    write_jsonl([AnnotationRecord(item_id="x", instances=[])], ann_raw, overwrite=True)
    cfg = Config(
        {
            "paths": {
                "annotation_path": str(ann_raw),
                "refined_annotation_path": str(tmp_path / "ann_refined.jsonl"),
                "masks_refined_dir": str(tmp_path / "masks_refined"),
            },
            "local_postprocess": {},
            "refine": {"report_path": str(tmp_path / "report.jsonl")},
        }
    )
    ann, report = refine_masks(cfg, project_root=tmp_path)
    recs = read_jsonl_list(ann, model=AnnotationRecord)
    assert recs[0].instances == []
    assert len(read_jsonl_list(report)) == 0