"""Tests for MatAnyone2Adapter, MattingAnythingAdapter, CascadePSPAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

from hmp.adapters.mask_refine import CascadePSPAdapter
from hmp.adapters.matting import MatAnyone2Adapter, MattingAnythingAdapter


def _mock(*out_files: str) -> list[str]:
    parts = ["import json,pathlib"]
    for f in out_files:
        if f.endswith(".json"):
            parts.append(f"pathlib.Path('{f}').write_text(json.dumps({{'q':0.9}}))")
        elif f.endswith(".png"):
            parts.append(f"pathlib.Path('{f}').write_bytes(b'\\x89PNG')")
        else:
            parts.append(f"pathlib.Path('{f}').mkdir(parents=True,exist_ok=True)")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# MatAnyone2Adapter
# ---------------------------------------------------------------------- #
def test_matanyone2_adapter_uses_registry_spec(tmp_path: Path):
    adapter = MatAnyone2Adapter(tmp_path)
    assert adapter.spec.name == "matanyone2"
    assert set(adapter.spec.expected_outputs) == {"alpha_video", "eval_map", "quality_score"}


def test_matanyone2_dry_run_resolves(tmp_path: Path):
    adapter = MatAnyone2Adapter(tmp_path)
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "/frames" in res.command and "target.png" in res.command
    assert "--eval-map" in res.command
    assert str(outputs["alpha_video"]).endswith("alpha.mp4")
    assert str(outputs["eval_map"]).endswith("eval_map")
    assert str(outputs["quality_score"]).endswith("quality_score.json")


def test_matanyone2_run_mock_creates_all_outputs(tmp_path: Path):
    adapter = MatAnyone2Adapter(
        tmp_path,
        command_template=_mock("{output_alpha_video}", "{output_eval_map}", "{output_quality_score}"),
    )
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_video"].exists()
    assert outputs["eval_map"].is_dir()
    assert outputs["quality_score"].exists()


def test_matanyone2_run_only_alpha_reports_missing(tmp_path: Path):
    adapter = MatAnyone2Adapter(tmp_path, command_template=_mock("{output_alpha_video}"))
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert set(res.missing_outputs) == {"eval_map", "quality_score"}
    assert res.ok is False


def test_matanyone2_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = MatAnyone2Adapter(tmp_path)
    res, _ = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"alpha_video", "eval_map", "quality_score"}


# ---------------------------------------------------------------------- #
# MattingAnythingAdapter
# ---------------------------------------------------------------------- #
def test_matting_anything_adapter_uses_registry_spec(tmp_path: Path):
    adapter = MattingAnythingAdapter(tmp_path)
    assert adapter.spec.name == "matting_anything"
    assert adapter.spec.expected_outputs == ["alpha_image"]


def test_matting_anything_dry_run_resolves(tmp_path: Path):
    adapter = MattingAnythingAdapter(tmp_path)
    res, outputs = adapter.mat("img.png", "mask.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command and "mask.png" in res.command
    assert str(outputs["alpha_image"]).endswith("alpha_image.png")


def test_matting_anything_run_mock_writes_alpha(tmp_path: Path):
    adapter = MattingAnythingAdapter(tmp_path, command_template=_mock("{output_alpha_image}"))
    res, outputs = adapter.mat("img.png", "mask.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["alpha_image"].exists()


def test_matting_anything_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = MattingAnythingAdapter(tmp_path)
    res, _ = adapter.mat("img.png", "mask.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.missing_outputs == ["alpha_image"]


# ---------------------------------------------------------------------- #
# CascadePSPAdapter
# ---------------------------------------------------------------------- #
def test_cascadepsp_adapter_uses_registry_spec(tmp_path: Path):
    adapter = CascadePSPAdapter(tmp_path)
    assert adapter.spec.name == "cascadepsp"
    assert adapter.spec.expected_outputs == ["refined_mask"]
    # priority 3
    assert adapter.spec.priority == 3


def test_cascadepsp_dry_run_resolves(tmp_path: Path):
    adapter = CascadePSPAdapter(tmp_path)
    res, outputs = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command and "coarse.png" in res.command
    assert str(outputs["refined_mask"]).endswith("refined_mask.png")


def test_cascadepsp_run_mock_writes_mask(tmp_path: Path):
    adapter = CascadePSPAdapter(tmp_path, command_template=_mock("{output_refined_mask}"))
    res, outputs = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["refined_mask"].exists()


def test_cascadepsp_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = CascadePSPAdapter(tmp_path)
    res, _ = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.missing_outputs == ["refined_mask"]


# ---------------------------------------------------------------------- #
# Registry drift sanity
# ---------------------------------------------------------------------- #
def test_remaining_adapter_output_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(MatAnyone2Adapter(tmp_path).spec.expected_outputs) == set(reg.get("matanyone2").expected_outputs)
    assert set(MattingAnythingAdapter(tmp_path).spec.expected_outputs) == set(reg.get("matting_anything").expected_outputs)
    assert set(CascadePSPAdapter(tmp_path).spec.expected_outputs) == set(reg.get("cascadepsp").expected_outputs)