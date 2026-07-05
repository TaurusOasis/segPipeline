"""End-to-end test for the full 12-stage relabel pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from typer.testing import CliRunner

from hmp.cli import app
from hmp.common.jsonl import read_jsonl_list
from hmp.schemas import AlphaLabelRecord, RelabelTask

runner = CliRunner()


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "project": {"name": "relabel-test", "seed": 42},
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "annotation_path": str(tmp_path / "ann_raw.jsonl"),
            "refined_annotation_path": str(tmp_path / "ann_refined.jsonl"),
            "masks_raw_dir": str(tmp_path / "masks_raw"),
            "masks_refined_dir": str(tmp_path / "masks_refined"),
            "alpha_dir": str(tmp_path / "alpha"),
        },
        "manifest": {"media_type": "image", "tags": ["demo"]},
        "ingest": {"datasets": ["coco_rem"], "registry_path": str(Path(__file__).resolve().parents[1] / "configs/datasets.yaml")},
        "stratification": {"output_path": str(tmp_path / "manifest.jsonl")},
        "dummy": {"width_fraction": 0.5, "height_fraction": 0.7},
        "local_postprocess": {"remove_small_components": True, "min_component_area": 8, "fill_holes": True},
        "refine": {"report_path": str(tmp_path / "refine_report.jsonl")},
        "adaptive_trimap": {
            "output_dir": str(tmp_path / "alpha/adaptive_trimaps"),
            "roi_dir": str(tmp_path / "alpha/roi"),
            "base_radius": 4,
            "max_radius": 10,
        },
        "relabel": {
            "queue_path": str(tmp_path / "alpha/relabel_queue.jsonl"),
            "fused_alpha_dir": str(tmp_path / "alpha/fused"),
            "bbox_output_dir": str(tmp_path / "alpha/bboxes"),
            "eval_map_dir": str(tmp_path / "alpha/eval_maps"),
            "hitl_queue_path": str(tmp_path / "alpha/hitl_queue.jsonl"),
            "labels_path": str(tmp_path / "alpha/alpha_labels.jsonl"),
        },
    }
    import yaml

    p = tmp_path / "demo_relabel.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _fixtures(raw: Path, n: int = 4) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 96), (40 + i, 60, 80))
        d = ImageDraw.Draw(img)
        d.rectangle([16, 20, 48, 80], fill=(60, 90, 160))
        img.save(raw / f"demo_{i:02d}.jpg")


def test_full_relabel_pipeline(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    import yaml

    raw = Path(yaml.safe_load(cfg_path.read_text())["paths"]["raw_dir"])
    _fixtures(raw, n=4)

    result = runner.invoke(app, ["pipeline", "run-relabel", "--config", str(cfg_path), "--provider", "mock"])
    assert result.exit_code == 0, result.output

    labels_path = tmp_path / "alpha" / "alpha_labels.jsonl"
    assert labels_path.exists()
    labels = read_jsonl_list(labels_path, model=AlphaLabelRecord)
    assert len(labels) == 4
    assert all(label.alpha_path for label in labels)

    queue = read_jsonl_list(tmp_path / "alpha" / "relabel_queue.jsonl", model=RelabelTask)
    assert len(queue) == 4
    assert all(len(task.steps) == 12 for task in queue)
    assert (tmp_path / "alpha" / "fused").exists()
    assert list((tmp_path / "alpha" / "fused").glob("*_alpha.png"))
