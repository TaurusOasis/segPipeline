"""Tests for GroundedSam2Adapter, GroundingDinoAdapter, YoloDetectAdapter, MmagicMetricsAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

from hmp.adapters.detection import (
    GroundingDinoAdapter,
    GroundedSam2Adapter,
    YoloDetectAdapter,
)
from hmp.adapters.qa import MmagicMetricsAdapter


def _mock(*out_files: str) -> list[str]:
    parts = ["import json,pathlib"]
    for f in out_files:
        if f.endswith(".json"):
            parts.append(f"pathlib.Path('{f}').write_text(json.dumps({{'v':0.9}}))")
        else:
            parts.append(f"pathlib.Path('{f}').mkdir(parents=True,exist_ok=True)")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# GroundedSam2Adapter
# ---------------------------------------------------------------------- #
def test_grounded_sam2_uses_registry_spec(tmp_path: Path):
    adapter = GroundedSam2Adapter(tmp_path)
    assert adapter.spec.name == "grounded_sam2"
    assert set(adapter.spec.expected_outputs) == {"person_candidates", "bbox", "rle_mask", "score"}
    assert adapter.spec.priority == 1


def test_grounded_sam2_dry_run_resolves(tmp_path: Path):
    adapter = GroundedSam2Adapter(tmp_path, text_prompt="person . hair .")
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command
    assert "person . hair ." in res.command
    assert "--output-bbox" in res.command and "--output-rle" in res.command
    assert str(outputs["person_candidates"]).endswith("person_candidates.json")
    assert str(outputs["rle_mask"]).endswith("rle_mask.json")


def test_grounded_sam2_run_mock_creates_all(tmp_path: Path):
    adapter = GroundedSam2Adapter(
        tmp_path,
        command_template=_mock(
            "{output_person_candidates}", "{output_bbox}",
            "{output_rle_mask}", "{output_score}",
        ),
    )
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    for p in outputs.values():
        assert p.exists()


def test_grounded_sam2_run_partial_reports_missing(tmp_path: Path):
    adapter = GroundedSam2Adapter(
        tmp_path, command_template=_mock("{output_person_candidates}", "{output_bbox}")
    )
    res, _ = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert set(res.missing_outputs) == {"rle_mask", "score"}
    assert res.ok is False


def test_grounded_sam2_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = GroundedSam2Adapter(tmp_path)
    res, _ = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"person_candidates", "bbox", "rle_mask", "score"}


# ---------------------------------------------------------------------- #
# GroundingDinoAdapter
# ---------------------------------------------------------------------- #
def test_groundingdino_uses_registry_spec(tmp_path: Path):
    adapter = GroundingDinoAdapter(tmp_path)
    assert adapter.spec.name == "groundingdino"
    assert set(adapter.spec.expected_outputs) == {"bbox", "score", "phrase"}
    assert adapter.spec.priority == 2


def test_groundingdino_dry_run_resolves(tmp_path: Path):
    adapter = GroundingDinoAdapter(tmp_path, text_prompt="person")
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "--output-phrase" in res.command
    assert str(outputs["phrase"]).endswith("phrase.json")


def test_groundingdino_run_mock_creates_all(tmp_path: Path):
    adapter = GroundingDinoAdapter(
        tmp_path, command_template=_mock("{output_bbox}", "{output_score}", "{output_phrase}")
    )
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    for p in outputs.values():
        assert p.exists()


def test_groundingdino_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = GroundingDinoAdapter(tmp_path)
    res, _ = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"bbox", "score", "phrase"}


# ---------------------------------------------------------------------- #
# YoloDetectAdapter
# ---------------------------------------------------------------------- #
def test_yolo_uses_registry_spec(tmp_path: Path):
    adapter = YoloDetectAdapter(tmp_path)
    assert adapter.spec.name == "ultralytics_yolo"
    assert set(adapter.spec.expected_outputs) == {"bbox", "mask", "score"}
    assert adapter.spec.priority == 2


def test_yolo_dry_run_resolves(tmp_path: Path):
    adapter = YoloDetectAdapter(tmp_path, weights="yolo26s-seg.pt")
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "yolo26s-seg.pt" in res.command
    assert "--output-mask" in res.command
    assert str(outputs["mask"]).endswith("mask.json")


def test_yolo_run_mock_creates_all(tmp_path: Path):
    adapter = YoloDetectAdapter(
        tmp_path, command_template=_mock("{output_bbox}", "{output_mask}", "{output_score}")
    )
    res, outputs = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    for p in outputs.values():
        assert p.exists()


def test_yolo_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = YoloDetectAdapter(tmp_path)
    res, _ = adapter.detect("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"bbox", "mask", "score"}


# ---------------------------------------------------------------------- #
# MmagicMetricsAdapter
# ---------------------------------------------------------------------- #
def test_mmagic_uses_registry_spec(tmp_path: Path):
    adapter = MmagicMetricsAdapter(tmp_path)
    assert adapter.spec.name == "mmagic"
    assert set(adapter.spec.expected_outputs) == {"sad", "mse", "gradient", "connectivity"}


def test_mmagic_dry_run_resolves(tmp_path: Path):
    adapter = MmagicMetricsAdapter(tmp_path)
    res, outputs = adapter.metrics("pred/", "gt/", "trimap/", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "pred/" in res.command and "gt/" in res.command and "trimap/" in res.command
    assert "--output-sad" in res.command and "--output-connectivity" in res.command
    assert str(outputs["sad"]).endswith("sad.json")
    assert str(outputs["connectivity"]).endswith("connectivity.json")


def test_mmagic_run_mock_creates_all(tmp_path: Path):
    adapter = MmagicMetricsAdapter(
        tmp_path,
        command_template=_mock(
            "{output_sad}", "{output_mse}", "{output_gradient}", "{output_connectivity}"
        ),
    )
    res, outputs = adapter.metrics("pred/", "gt/", "trimap/", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    for p in outputs.values():
        assert p.exists()


def test_mmagic_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = MmagicMetricsAdapter(tmp_path)
    res, _ = adapter.metrics("pred/", "gt/", "trimap/", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"sad", "mse", "gradient", "connectivity"}


# ---------------------------------------------------------------------- #
# Registry drift sanity
# ---------------------------------------------------------------------- #
def test_detection_and_mmagic_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(GroundedSam2Adapter(tmp_path).spec.expected_outputs) == set(reg.get("grounded_sam2").expected_outputs)
    assert set(GroundingDinoAdapter(tmp_path).spec.expected_outputs) == set(reg.get("groundingdino").expected_outputs)
    assert set(YoloDetectAdapter(tmp_path).spec.expected_outputs) == set(reg.get("ultralytics_yolo").expected_outputs)
    assert set(MmagicMetricsAdapter(tmp_path).spec.expected_outputs) == set(reg.get("mmagic").expected_outputs)