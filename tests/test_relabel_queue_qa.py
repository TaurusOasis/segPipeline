"""Tests for relabel queue QA decision propagation."""

from __future__ import annotations

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config
from hmp.data.mask_io import write_binary_mask
from hmp.matting.relabel_queue import build_relabel_queue
from hmp.schemas import AnnotationRecord, InstanceAnnotation, MediaItem, RelabelTask


def _base_cfg(tmp_path):
    raw = tmp_path / "raw"
    mask_dir = tmp_path / "masks_refined"
    trimap_dir = tmp_path / "alpha" / "trimaps"
    raw.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    trimap_dir.mkdir(parents=True)

    image = raw / "img.jpg"
    image.write_bytes(b"placeholder")
    mask = mask_dir / "img_person_0.png"
    trimap = trimap_dir / "img_person_0_trimap.png"
    import numpy as np

    write_binary_mask(mask, np.ones((8, 8), dtype=bool))
    trimap.write_bytes(b"placeholder")

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            MediaItem(
                item_id="img",
                media_type="image",
                path=str(image),
                width=8,
                height=8,
                sha256="sha",
            )
        ],
    )
    return {
        "paths": {
            "manifest_path": str(manifest),
            "refined_annotation_path": str(tmp_path / "annotations_refined.jsonl"),
            "alpha_dir": str(tmp_path / "alpha"),
        },
        "trimap": {"output_dir": str(trimap_dir)},
        "relabel": {"queue_path": str(tmp_path / "alpha" / "relabel_queue.jsonl")},
    }


def test_relabel_queue_accept_skips_review(tmp_path):
    cfg_dict = _base_cfg(tmp_path)
    write_jsonl(
        tmp_path / "annotations_refined.jsonl",
        [
            AnnotationRecord(
                item_id="img",
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[1, 1, 7, 7],
                        mask_path=str(tmp_path / "masks_refined" / "img_person_0.png"),
                        prompt_history=[
                            {
                                "decision": "accept",
                                "quality_scores": {"semantic_score": 0.92, "boundary_score": 0.90},
                            }
                        ],
                    )
                ],
            )
        ],
    )
    out = build_relabel_queue(Config(cfg_dict), project_root=tmp_path)
    task = read_jsonl_list(out, model=RelabelTask)[0]
    assert task.review_required is False
    assert task.status == "ready"
    assert task.quality_scores["semantic_score"] == 0.92


def test_relabel_queue_reject_marks_rejected(tmp_path):
    cfg_dict = _base_cfg(tmp_path)
    write_jsonl(
        tmp_path / "annotations_refined.jsonl",
        [
            AnnotationRecord(
                item_id="img",
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[1, 1, 7, 7],
                        mask_path=str(tmp_path / "masks_refined" / "img_person_0.png"),
                        prompt_history=[{"decision": "reject", "error_tags": ["low_iou"]}],
                    )
                ],
            )
        ],
    )
    out = build_relabel_queue(Config(cfg_dict), project_root=tmp_path)
    task = read_jsonl_list(out, model=RelabelTask)[0]
    assert task.review_required is True
    assert task.status == "rejected"
    assert task.prompt_history[0]["error_tags"] == ["low_iou"]
