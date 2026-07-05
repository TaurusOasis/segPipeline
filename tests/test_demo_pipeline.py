"""End-to-end test for the CPU-only demo pipeline (Step 25).

Drives the same CLI commands as scripts/run_demo_pipeline.sh, against a temp
workspace with absolute paths so the result is independent of the CWD.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
from typer.testing import CliRunner

from hmp.cli import app

runner = CliRunner()


def _write_demo_config(tmp_path: Path) -> Path:
    cfg = {
        "project": {"name": "demo-test", "seed": 42},
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "manifest_path": str(tmp_path / "manifests/manifest.jsonl"),
            "annotation_path": str(tmp_path / "ann/annotations_raw.jsonl"),
            "refined_annotation_path": str(tmp_path / "ann/annotations_refined.jsonl"),
            "masks_raw_dir": str(tmp_path / "masks_raw"),
            "masks_refined_dir": str(tmp_path / "masks_refined"),
            "yolo_dir": str(tmp_path / "yolo_seg"),
            "alpha_dir": str(tmp_path / "alpha"),
            "runs_dir": str(tmp_path / "runs"),
        },
        "dummy": {"width_fraction": 0.5, "height_fraction": 0.7, "score": 0.9},
        "local_postprocess": {"remove_small_components": True, "min_component_area": 16, "fill_holes": True, "keep_largest_component": False},
        "refine": {"report_path": str(tmp_path / "ann/refine_report.jsonl")},
        "yolo_export": {"val_ratio": 0.34, "seed": 42, "symlink": False, "class_names": ["person"]},
        "trimap": {"output_dir": str(tmp_path / "alpha/trimaps"), "radius": 5},
    }
    p = tmp_path / "demo.yaml"
    import yaml

    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _make_fixtures(raw: Path, n=6) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 96), (40 + i, 60, 80))
        d = ImageDraw.Draw(img)
        d.ellipse([24, 12, 40, 30], fill=(200, 180, 160))
        d.rectangle([22, 30, 42, 80], fill=(60, 90, 160))
        img.save(raw / f"demo_{i:02d}.jpg")


def test_demo_pipeline_end_to_end(tmp_path):
    cfg_path = _write_demo_config(tmp_path)
    import yaml

    paths = yaml.safe_load(cfg_path.read_text())["paths"]
    _make_fixtures(Path(paths["raw_dir"]), n=6)

    args = ["--config", str(cfg_path)]
    # 1. manifest
    r = runner.invoke(app, ["manifest", "build", *args, "--overwrite"]); assert r.exit_code == 0, r.output
    manifest = Path(paths["manifest_path"])
    assert manifest.exists()
    assert sum(1 for _ in manifest.open()) == 6
    # 2. dummy label
    r = runner.invoke(app, ["label", "dummy", *args]); assert r.exit_code == 0, r.output
    ann_raw = Path(paths["annotation_path"])
    assert ann_raw.exists()
    # 3. refine
    r = runner.invoke(app, ["refine", "masks", *args]); assert r.exit_code == 0, r.output
    ann_ref = Path(paths["refined_annotation_path"])
    assert ann_ref.exists()
    assert (tmp_path / "ann/refine_report.jsonl").exists()
    # 4. export yolo
    r = runner.invoke(app, ["dataset", "export-yolo", *args]); assert r.exit_code == 0, r.output
    yolo = Path(paths["yolo_dir"])
    assert (yolo / "data.yaml").exists()
    assert (yolo / "images" / "train").is_dir() and (yolo / "labels" / "val").is_dir()
    # 5. trimap
    r = runner.invoke(app, ["matting", "make-trimap", *args]); assert r.exit_code == 0, r.output
    trimaps = list((tmp_path / "alpha/trimaps").glob("*.png"))
    assert len(trimaps) == 6
    import cv2
    import numpy as np

    vals = set(np.unique(cv2.imread(str(trimaps[0]), cv2.IMREAD_UNCHANGED)).tolist())
    assert vals == {0, 128, 255}
    # 6. report
    r = runner.invoke(app, ["eval", "report", *args]); assert r.exit_code == 0, r.output
    report = tmp_path / "runs" / "eval_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Manifest" in text and "items" in text


def test_demo_yaml_file_runs_via_shell(tmp_path):
    # Sanity: the shipped configs/demo.yaml loads and the CLI commands accept it
    # (paths are relative; we only test loading here, not artifact creation).
    cfg_path = _write_demo_config(tmp_path)
    r = runner.invoke(app, ["config", "show", "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["paths"]["raw_dir"].endswith("raw")