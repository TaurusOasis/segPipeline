"""Tests for hmp.schemas and hmp.common.jsonl (Step 01)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hmp.common.jsonl import count_jsonl, read_jsonl, read_jsonl_list, write_jsonl
from hmp.schemas import (
    AlphaLabelRecord,
    AnnotationRecord,
    InstanceAnnotation,
    MediaItem,
    QualityRecord,
    RelabelStep,
    RelabelTask,
    validate_bbox_xyxy,
)


# ---------------------------------------------------------------------------
# MediaItem
# ---------------------------------------------------------------------------
def test_media_item_valid():
    m = MediaItem(
        item_id="img001",
        media_type="image",
        path="data/raw/img001.jpg",
        width=1920,
        height=1080,
        sha256="abc",
        tags=["raw"],
    )
    assert m.item_id == "img001"
    assert m.width == 1920


def test_media_item_rejects_nonpositive_dims():
    with pytest.raises(ValidationError):
        MediaItem(item_id="x", path="p", width=0, height=10, sha256="a")
    with pytest.raises(ValidationError):
        MediaItem(item_id="x", path="p", width=10, height=-1, sha256="a")


def test_media_item_frame_fields():
    m = MediaItem(
        item_id="v001_f000123",
        media_type="frame",
        path="data/frames/v001/frame_000123.jpg",
        width=1280,
        height=720,
        sha256="a",
        source_video="data/raw/v001.mp4",
        frame_index=123,
        timestamp_ms=4100,
        tags=["video_frame"],
    )
    assert m.frame_index == 123
    assert m.timestamp_ms == 4100


# ---------------------------------------------------------------------------
# InstanceAnnotation / bbox validation
# ---------------------------------------------------------------------------
def test_instance_valid_bbox():
    inst = InstanceAnnotation(instance_id="person_0", bbox_xyxy=[10, 10, 100, 200])
    assert inst.bbox_xyxy == [10, 10, 100, 200]
    assert inst.category == "person"


def test_instance_rejects_bad_bbox_order():
    with pytest.raises(ValidationError):
        InstanceAnnotation(instance_id="p", bbox_xyxy=[100, 10, 10, 200])  # x2<=x1
    with pytest.raises(ValidationError):
        InstanceAnnotation(instance_id="p", bbox_xyxy=[10, 200, 100, 10])  # y2<=y1


def test_instance_rejects_negative_bbox():
    with pytest.raises(ValidationError):
        InstanceAnnotation(instance_id="p", bbox_xyxy=[-1, 10, 100, 200])


def test_instance_score_bounds():
    inst = InstanceAnnotation(instance_id="p", bbox_xyxy=[10, 10, 20, 20], score=0.5)
    assert inst.score == 0.5
    with pytest.raises(ValidationError):
        InstanceAnnotation(instance_id="p", bbox_xyxy=[10, 10, 20, 20], score=1.5)


def test_validate_bbox_xyxy_standalone():
    assert validate_bbox_xyxy([0, 0, 5, 5]) == [0, 0, 5, 5]
    with pytest.raises(ValueError):
        validate_bbox_xyxy([5, 0, 5, 5])


# ---------------------------------------------------------------------------
# AnnotationRecord / QualityRecord
# ---------------------------------------------------------------------------
def test_annotation_record_empty_instances():
    rec = AnnotationRecord(item_id="img001")
    assert rec.instances == []


def test_annotation_record_with_instances():
    rec = AnnotationRecord(
        item_id="img001",
        instances=[
            InstanceAnnotation(instance_id="person_0", bbox_xyxy=[1, 1, 2, 2]),
        ],
    )
    assert len(rec.instances) == 1


def test_quality_record_decision():
    q = QualityRecord(item_id="x", scores={"blur_score": 0.1}, decision="keep", reason="ok")
    assert q.decision == "keep"
    with pytest.raises(ValidationError):
        QualityRecord(item_id="x", decision="nope")


def test_relabel_task_schema():
    task = RelabelTask(
        task_id="img_person_0",
        item_id="img",
        instance_id="person_0",
        media_type="frame",
        image_path="image.jpg",
        source_video="video.mp4",
        frame_index=3,
        mask_path="mask.png",
        masklet_path="masklet.json",
        trimap_path="trimap.png",
        trimap_or_roi_path="trimap.png",
        alpha_path="alpha.png",
        alpha_exr_path="alpha.exr",
        eval_map_path="eval_map.png",
        bbox_path="bbox.json",
        bbox_xyxy=[1, 2, 10, 20],
        target_id="target_a",
        keypoints_path="keypoints.json",
        expected_outputs={"alpha": "alpha.png"},
        steps=[RelabelStep(index=0, name="data_source_sampling", status="done")],
        prompt_history=[{"prompt_type": "box"}],
        branch_source={"Bv": "alpha_bv.png", "Bi": None},
        license_meta={"license": "research-only", "source_url": "https://example.test"},
        status="ready",
    )
    assert task.review_required is True
    assert task.steps[0].name == "data_source_sampling"
    assert task.media_type == "frame"
    assert task.source_video == "video.mp4"
    assert task.trimap_or_roi_path == "trimap.png"
    assert task.branch_source["Bi"] is None
    assert task.license_meta["license"] == "research-only"
    with pytest.raises(ValidationError):
        RelabelTask(
            task_id="bad",
            item_id="img",
            instance_id="person_0",
            alpha_path="alpha.png",
            bbox_path="bbox.json",
            bbox_xyxy=[10, 2, 1, 20],
        )


def test_alpha_label_record_schema():
    label = AlphaLabelRecord(
        item_id="img",
        instance_id="person_0",
        image_path="image.jpg",
        source_video="video.mp4",
        frame_index=3,
        alpha_path="alpha.png",
        alpha_exr_path="alpha.exr",
        mask_path="mask.png",
        masklet_path="masklet.json",
        trimap_path="trimap.png",
        trimap_or_roi_path="trimap.png",
        eval_map_path="eval_map.png",
        bbox_path="bbox.json",
        bbox_xyxy=[1, 2, 10, 20],
        target_id="target_a",
        keypoints_path="keypoints.json",
        branch_source={"Bv": "alpha_bv.png", "Bg": None},
        prompt_history=[{"prompt_type": "mask"}],
        license_meta={"license": "research-only"},
        quality_score=0.8,
        review_status="accepted",
    )
    assert label.quality_score == 0.8
    assert label.eval_map_path == "eval_map.png"
    assert label.branch_source["Bg"] is None
    with pytest.raises(ValidationError):
        AlphaLabelRecord(
            item_id="img",
            instance_id="person_0",
            image_path="image.jpg",
            alpha_path="alpha.png",
            bbox_xyxy=[1, 2, 10, 20],
            review_status="maybe",
        )


# ---------------------------------------------------------------------------
# JSONL round-trips
# ---------------------------------------------------------------------------
def test_write_then_read_manifest(tmp_path):
    p = tmp_path / "manifest.jsonl"
    items = [
        MediaItem(item_id="a", path="a.jpg", width=10, height=10, sha256="aa"),
        MediaItem(item_id="b", path="b.jpg", width=20, height=20, sha256="bb"),
    ]
    write_jsonl(p, items)
    assert count_jsonl(p) == 2
    out = read_jsonl_list(p, model=MediaItem)
    assert [m.item_id for m in out] == ["a", "b"]
    assert out[1].width == 20


def test_write_then_read_annotations(tmp_path):
    p = tmp_path / "ann.jsonl"
    recs = [
        AnnotationRecord(item_id="a", instances=[]),
        AnnotationRecord(
            item_id="b",
            instances=[InstanceAnnotation(instance_id="p0", bbox_xyxy=[1, 1, 9, 9], score=0.9)],
        ),
    ]
    write_jsonl(p, recs)
    out = read_jsonl_list(p, model=AnnotationRecord)
    assert out[0].instances == []
    assert out[1].instances[0].bbox_xyxy == [1, 1, 9, 9]


def test_write_then_read_quality(tmp_path):
    p = tmp_path / "q.jsonl"
    recs = [
        QualityRecord(item_id="a", scores={"blur_score": 0.2}, decision="keep"),
        QualityRecord(item_id="b", scores={"blur_score": 0.9}, decision="drop", reason="blurry"),
    ]
    write_jsonl(p, recs)
    out = read_jsonl_list(p, model=QualityRecord)
    assert out[1].decision == "drop"
    assert out[1].reason == "blurry"


def test_write_jsonl_no_overwrite(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [MediaItem(item_id="a", path="a", width=1, height=1, sha256="x")])
    with pytest.raises(FileExistsError):
        write_jsonl(p, [MediaItem(item_id="a", path="a", width=1, height=1, sha256="x")], overwrite=False)


def test_write_jsonl_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "dir" / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    assert p.exists()


def test_read_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_jsonl(tmp_path / "nope.jsonl"))


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    out = read_jsonl_list(p)
    assert out == [{"a": 1}, {"a": 2}]


def test_read_jsonl_invalid_json(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl_list(p)
