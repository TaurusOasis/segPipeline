"""Tests for VideoMaMaAdapter (Bd), DiffMatteAdapter, SDMatteAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hmp.adapters.diffusion import DiffMatteAdapter, SDMatteAdapter, VideoMaMaAdapter


def _mock(*out_files: str) -> list[str]:
    parts = ["import json,pathlib"]
    for f in out_files:
        if f.endswith(".json"):
            parts.append(f"pathlib.Path('{f}').write_text(json.dumps({{'roi':'r'}}))")
        elif f.endswith(".png"):
            parts.append(f"pathlib.Path('{f}').write_bytes(b'\\x89PNG')")
        else:
            parts.append(f"pathlib.Path('{f}').mkdir(parents=True,exist_ok=True)")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# VideoMaMaAdapter
# ---------------------------------------------------------------------- #
def test_videomama_adapter_uses_registry_spec(tmp_path: Path):
    adapter = VideoMaMaAdapter(tmp_path)
    assert adapter.spec.name == "videomama"
    assert set(adapter.spec.expected_outputs) == {"alpha_diffusion", "refine_roi"}
    assert adapter.spec.license_review == "non_commercial_risk"


def test_videomama_dry_run_resolves(tmp_path: Path):
    adapter = VideoMaMaAdapter(tmp_path)
    res, outputs = adapter.refine("/frames", "/coarse", "10,10,40,50", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "10,10,40,50" in res.command
    assert "--refine-roi" in res.command
    assert str(outputs["alpha_diffusion"]).endswith("alpha_diffusion")
    assert str(outputs["refine_roi"]).endswith("refine_roi.json")


def test_videomama_run_mock_creates_outputs(tmp_path: Path):
    adapter = VideoMaMaAdapter(
        tmp_path,
        command_template=_mock("{output_alpha_diffusion}", "{output_refine_roi}"),
    )
    res, outputs = adapter.refine("/frames", "/coarse", "10,10,40,50", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_diffusion"].is_dir()
    assert outputs["refine_roi"].exists()


def test_videomama_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = VideoMaMaAdapter(tmp_path)
    res, _ = adapter.refine("/frames", "/coarse", "10,10,40,50", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert set(res.missing_outputs) == {"alpha_diffusion", "refine_roi"}


# ---------------------------------------------------------------------- #
# DiffMatteAdapter
# ---------------------------------------------------------------------- #
def test_diffmatte_adapter_uses_registry_spec(tmp_path: Path):
    adapter = DiffMatteAdapter(tmp_path)
    assert adapter.spec.name == "diffmatte"
    assert adapter.spec.expected_outputs == ["alpha_diffusion"]


def test_diffmatte_dry_run_resolves(tmp_path: Path):
    adapter = DiffMatteAdapter(tmp_path)
    res, outputs = adapter.refine("img.png", "mask.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command and "mask.png" in res.command
    assert str(outputs["alpha_diffusion"]).endswith("alpha_diffusion.png")


def test_diffmatte_run_mock_writes_alpha(tmp_path: Path):
    adapter = DiffMatteAdapter(tmp_path, command_template=_mock("{output_alpha_diffusion}"))
    res, outputs = adapter.refine("img.png", "mask.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_diffusion"].exists()


def test_diffmatte_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = DiffMatteAdapter(tmp_path)
    res, _ = adapter.refine("img.png", "mask.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.missing_outputs == ["alpha_diffusion"]


# ---------------------------------------------------------------------- #
# SDMatteAdapter
# ---------------------------------------------------------------------- #
def test_sdmatte_adapter_uses_registry_spec(tmp_path: Path):
    adapter = SDMatteAdapter(tmp_path)
    assert adapter.spec.name == "sdmatte"
    assert adapter.spec.expected_outputs == ["alpha_diffusion"]


def test_sdmatte_dry_run_resolves(tmp_path: Path):
    adapter = SDMatteAdapter(tmp_path)
    res, outputs = adapter.refine("img.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command
    assert str(outputs["alpha_diffusion"]).endswith("alpha_diffusion.png")


def test_sdmatte_run_mock_writes_alpha(tmp_path: Path):
    adapter = SDMatteAdapter(tmp_path, command_template=_mock("{output_alpha_diffusion}"))
    res, outputs = adapter.refine("img.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["alpha_diffusion"].exists()


# ---------------------------------------------------------------------- #
# Registry drift sanity
# ---------------------------------------------------------------------- #
def test_diffusion_adapter_output_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(VideoMaMaAdapter(tmp_path).spec.expected_outputs) == set(reg.get("videomama").expected_outputs)
    assert set(DiffMatteAdapter(tmp_path).spec.expected_outputs) == set(reg.get("diffmatte").expected_outputs)
    assert set(SDMatteAdapter(tmp_path).spec.expected_outputs) == set(reg.get("sdmatte").expected_outputs)