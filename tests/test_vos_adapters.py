"""Tests for CutieAdapter and XMemAdapter (VOS masklet propagation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hmp.adapters.vos import CutieAdapter, XMemAdapter


def _mock_make_dir_and_json(*out_files: str) -> list[str]:
    """Mock command: create the listed dir/json outputs."""
    parts = ["import json,pathlib"]
    for f in out_files:
        if f.endswith(".json"):
            parts.append(f"pathlib.Path('{f}').write_text(json.dumps({{'track':'t1'}}))")
        else:
            parts.append(f"pathlib.Path('{f}').mkdir(parents=True,exist_ok=True)")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# CutieAdapter
# ---------------------------------------------------------------------- #
def test_cutie_adapter_uses_registry_spec(tmp_path: Path):
    adapter = CutieAdapter(tmp_path)
    assert adapter.spec.name == "cutie"
    assert set(adapter.spec.expected_outputs) == {"masklet", "track_id"}


def test_cutie_dry_run_resolves(tmp_path: Path):
    adapter = CutieAdapter(tmp_path)
    res, outputs = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "/frames" in res.command and "prompt.json" in res.command
    assert str(outputs["masklet"]).endswith("masklet")
    assert str(outputs["track_id"]).endswith("track_id.json")
    # Dry run does not create the masklet dir.
    assert not outputs["masklet"].exists()


def test_cutie_run_mock_creates_masklet_and_track_id(tmp_path: Path):
    adapter = CutieAdapter(
        tmp_path,
        command_template=_mock_make_dir_and_json("{output_masklet}", "{output_track_id}"),
    )
    res, outputs = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["masklet"].is_dir()
    assert outputs["track_id"].exists()


def test_cutie_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = CutieAdapter(tmp_path)
    res, _ = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert set(res.missing_outputs) == {"masklet", "track_id"}


def test_cutie_run_only_track_id_reports_masklet_missing(tmp_path: Path):
    adapter = CutieAdapter(tmp_path, command_template=_mock_make_dir_and_json("{output_track_id}"))
    res, outputs = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == ["masklet"]
    assert res.ok is False
    assert outputs["track_id"].exists()
    assert not outputs["masklet"].exists()


# ---------------------------------------------------------------------- #
# XMemAdapter
# ---------------------------------------------------------------------- #
def test_xmem_adapter_uses_registry_spec(tmp_path: Path):
    adapter = XMemAdapter(tmp_path)
    assert adapter.spec.name == "xmem"
    assert set(adapter.spec.expected_outputs) == {"masklet", "track_id"}


def test_xmem_dry_run_resolves(tmp_path: Path):
    adapter = XMemAdapter(tmp_path)
    res, outputs = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "xmem.infer" in " ".join(res.command)
    assert str(outputs["masklet"]).endswith("masklet")


def test_xmem_run_mock_creates_outputs(tmp_path: Path):
    adapter = XMemAdapter(
        tmp_path,
        command_template=_mock_make_dir_and_json("{output_masklet}", "{output_track_id}"),
    )
    res, outputs = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["masklet"].is_dir()
    assert outputs["track_id"].exists()


def test_xmem_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = XMemAdapter(tmp_path)
    res, _ = adapter.propagate("/frames", "prompt.json", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert set(res.missing_outputs) == {"masklet", "track_id"}


# ---------------------------------------------------------------------- #
# Registry drift sanity
# ---------------------------------------------------------------------- #
def test_vos_adapter_output_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(CutieAdapter(tmp_path).spec.expected_outputs) == set(reg.get("cutie").expected_outputs)
    assert set(XMemAdapter(tmp_path).spec.expected_outputs) == set(reg.get("xmem").expected_outputs)