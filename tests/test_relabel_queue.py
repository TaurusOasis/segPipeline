"""Tests for mask-to-matte relabeling queues."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from hmp.cli import app
from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config
from hmp.data.mask_io import write_binary_mask
from hmp.matting.relabel_queue import build_relabel_queue
from hmp.schemas import AnnotationRecord, InstanceAnnotation, MediaItem, RelabelTask


runner = CliRunner()


def _setup(tmp_path):
    raw = tmp_path / "raw"
    mask_dir = tmp_path / "masks_refined"
    trimap_dir = tmp_path / "alpha" / "trimaps"
    raw.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    trimap_dir.mkdir(parents=True)

    image = raw / "img.jpg"
    image.write_bytes(b"not-a-real-image-needed-for-this-test")
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
                media_type="frame",
                path=str(image),
                width=8,
                height=8,
                sha256="sha",
                source_video="video.mp4",
                frame_index=12,
                timestamp_ms=400,
                tags=["coco_rem"],
                license="research-only",
                source_url="https://example.test/source",
            )
        ],
    )
    ann = tmp_path / "annotations_refined.jsonl"
    write_jsonl(
        ann,
        [
            AnnotationRecord(
                item_id="img",
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[1, 1, 7, 7],
                        mask_path=str(mask),
                        track_id="track_1",
                        target_id="target_a",
                        keypoints_path=str(tmp_path / "keypoints.json"),
                    )
                ],
            )
        ],
    )
    cfg = Config(
        {
            "paths": {
                "manifest_path": str(manifest),
                "refined_annotation_path": str(ann),
                "alpha_dir": str(tmp_path / "alpha"),
            },
            "trimap": {"output_dir": str(trimap_dir)},
            "relabel": {"queue_path": str(tmp_path / "alpha" / "relabel_queue.jsonl")},
        }
    )
    return cfg


def test_build_relabel_queue_writes_tasks_and_bbox_sidecars(tmp_path):
    cfg = _setup(tmp_path)
    out = build_relabel_queue(cfg, project_root=tmp_path)
    tasks = read_jsonl_list(out, model=RelabelTask)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "img_person_0"
    assert task.status == "ready"
    assert task.media_type == "frame"
    assert task.source_video == "video.mp4"
    assert task.frame_index == 12
    assert task.source_dataset == "coco_rem"
    assert task.target_id == "target_a"
    assert task.keypoints_path.endswith("keypoints.json")
    assert task.expected_outputs["alpha"].endswith("img_person_0_alpha.png")
    assert task.expected_outputs["alpha_png"].endswith("img_person_0_alpha.png")
    assert task.expected_outputs["alpha_exr"].endswith("img_person_0_alpha.exr")
    assert task.expected_outputs["bbox"].endswith("img_person_0.json")
    assert task.expected_outputs["masklet"].endswith("img_person_0_masklet.json")
    assert task.expected_outputs["eval_map"].endswith("img_person_0_eval.png")
    assert task.branch_outputs["Bv"].endswith("img_person_0_alpha.png")
    assert task.branch_source["Bs"].endswith("img_person_0.png")
    assert task.prompt_history[0]["prompt_type"] == "box"
    assert task.license_meta["license"] == "research-only"
    assert len(task.steps) == 12
    assert task.steps[0].name == "data_source_sampling"
    assert task.steps[6].name == "matting_critical_roi"
    assert task.steps[6].status == "done"

    bbox = json.loads((tmp_path / "alpha" / "bboxes" / "img_person_0.json").read_text())
    assert bbox["bbox_xyxy"] == [1, 1, 7, 7]
    assert bbox["video_track_id"] == "track_1"
    assert bbox["target_id"] == "target_a"
    assert bbox["license_meta"]["license"] == "research-only"


def test_build_relabel_queue_dry_run(tmp_path):
    cfg = _setup(tmp_path)
    out = build_relabel_queue(cfg, project_root=tmp_path, dry_run=True)
    assert out.name == "relabel_queue.jsonl"
    assert not out.exists()


def test_build_relabel_queue_dry_run_without_annotations(tmp_path):
    cfg = Config(
        {
            "paths": {
                "manifest_path": str(tmp_path / "missing_manifest.jsonl"),
                "refined_annotation_path": str(tmp_path / "missing_annotations.jsonl"),
                "alpha_dir": str(tmp_path / "alpha"),
            },
            "relabel": {"queue_path": str(tmp_path / "alpha" / "relabel_queue.jsonl")},
        }
    )
    out = build_relabel_queue(cfg, project_root=tmp_path, dry_run=True)
    assert out == tmp_path / "alpha" / "relabel_queue.jsonl"
    assert not out.exists()


def test_relabel_queue_cli(tmp_path):
    cfg = _setup(tmp_path)
    import yaml

    cfg_path = tmp_path / "relabel.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    result = runner.invoke(app, ["relabel", "queue", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "alpha" / "relabel_queue.jsonl").exists()
