"""Tests for SematAdapter (Bi), MaggieAdapter, RvmAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hmp.adapters.matting import MaggieAdapter, RvmAdapter, SematAdapter


def _mock_write(*out_files: str) -> list[str]:
    """Build a mock command that writes the given output placeholders to disk."""
    parts = ["import pathlib"]
    for f in out_files:
        parts.append(f"pathlib.Path('{f}').write_bytes(b'\\x89PNG')")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# SematAdapter
# ---------------------------------------------------------------------- #
def test_semat_adapter_uses_registry_spec(tmp_path: Path):
    adapter = SematAdapter(tmp_path)
    assert adapter.spec.name == "semat"
    assert adapter.spec.expected_outputs == ["alpha_image"]


def test_semat_dry_run_resolves(tmp_path: Path):
    adapter = SematAdapter(tmp_path)
    res, outputs = adapter.mat("img.png", "person.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "img.png" in res.command and "person.png" in res.command
    assert str(outputs["alpha_image"]).endswith("alpha_image.png")


def test_semat_run_mock_writes_alpha(tmp_path: Path):
    adapter = SematAdapter(tmp_path, command_template=_mock_write("{output_alpha_image}"))
    res, outputs = adapter.mat("img.png", "person.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_image"].exists()


def test_semat_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = SematAdapter(tmp_path)
    res, _ = adapter.mat("img.png", "person.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert res.missing_outputs == ["alpha_image"]


# ---------------------------------------------------------------------- #
# MaggieAdapter
# ---------------------------------------------------------------------- #
def test_maggie_adapter_uses_registry_spec(tmp_path: Path):
    adapter = MaggieAdapter(tmp_path)
    assert adapter.spec.name == "maggie"
    assert set(adapter.spec.expected_outputs) == {"alpha", "instance_alpha"}


def test_maggie_dry_run_resolves_both_outputs(tmp_path: Path):
    adapter = MaggieAdapter(tmp_path)
    res, outputs = adapter.mat("img.png", "inst.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "--instance-output" in res.command
    assert str(outputs["alpha"]).endswith("alpha.png")
    assert str(outputs["instance_alpha"]).endswith("instance_alpha.png")


def test_maggie_run_mock_writes_both_outputs(tmp_path: Path):
    adapter = MaggieAdapter(
        tmp_path,
        command_template=_mock_write("{output_alpha}", "{output_instance_alpha}"),
    )
    res, outputs = adapter.mat("img.png", "inst.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha"].exists()
    assert outputs["instance_alpha"].exists()


def test_maggie_run_only_one_output_reports_missing(tmp_path: Path):
    # Mock writes only alpha -> instance_alpha missing.
    adapter = MaggieAdapter(tmp_path, command_template=_mock_write("{output_alpha}"))
    res, outputs = adapter.mat("img.png", "inst.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == ["instance_alpha"]
    assert res.ok is False


# ---------------------------------------------------------------------- #
# RvmAdapter
# ---------------------------------------------------------------------- #
def test_rvm_adapter_uses_registry_spec(tmp_path: Path):
    adapter = RvmAdapter(tmp_path)
    assert adapter.spec.name == "rvm"
    assert adapter.spec.expected_outputs == ["alpha_video"]


def test_rvm_dry_run_resolves(tmp_path: Path):
    adapter = RvmAdapter(tmp_path)
    res, outputs = adapter.mat("/frames", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "/frames" in res.command
    assert str(outputs["alpha_video"]).endswith("alpha.mp4")


def test_rvm_run_mock_writes_alpha_video(tmp_path: Path):
    adapter = RvmAdapter(tmp_path, command_template=_mock_write("{output_alpha_video}"))
    res, outputs = adapter.mat("/frames", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_video"].exists()


def test_rvm_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = RvmAdapter(tmp_path)
    res, _ = adapter.mat("/frames", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert res.missing_outputs == ["alpha_video"]


# ---------------------------------------------------------------------- #
# Shared: registry expected_outputs drift (covered in templates test, but
# sanity-check the matting group here too)
# ---------------------------------------------------------------------- #
def test_matting_adapter_output_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(SematAdapter(tmp_path).spec.expected_outputs) == set(reg.get("semat").expected_outputs)
    assert set(MaggieAdapter(tmp_path).spec.expected_outputs) == set(reg.get("maggie").expected_outputs)
    assert set(RvmAdapter(tmp_path).spec.expected_outputs) == set(reg.get("rvm").expected_outputs)