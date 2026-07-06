"""Tests for the concrete mask-refine adapters (SamRefiner, SamHQ)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hmp.adapters.mask_refine import HqSamAdapter, SamRefinerAdapter


# A mock SAMRefiner command: write a 1x1 PNG marker + a quality JSON so the
# output validation passes. Uses the resolved {output_*} placeholders.
_MOCK_REFINE = [
    sys.executable, "-c",
    (
        "import json,pathlib; "
        "pathlib.Path('{output_refined_mask}').write_bytes(b'\\x89PNG'); "
        "pathlib.Path('{output_mask_quality}').write_text(json.dumps({{'boundary_f1':0.9}}))"
    ),
]
_MOCK_HQ = [
    sys.executable, "-c",
    "import pathlib; pathlib.Path('{output_refined_mask}').write_bytes(b'\\x89PNG')",
]


# ---------------------------------------------------------------------- #
# SamRefinerAdapter — construction
# ---------------------------------------------------------------------- #
def test_samrefiner_adapter_uses_registry_spec(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path, repo_python="/opt/venv/bin/python")
    assert adapter.spec.name == "samrefiner"
    assert adapter.spec.expected_outputs == ["refined_mask", "mask_quality"]
    assert adapter.repo_python == "/opt/venv/bin/python"
    assert adapter.env_overlay["REPO_PYTHON"] == "/opt/venv/bin/python"
    # Default template is the catalog one.
    assert "-m" in adapter.command_template
    assert "samrefiner.refine" in adapter.command_template


def test_samrefiner_adapter_command_template_override(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path, command_template=["echo", "{input_image}"])
    assert adapter.command_template == ["echo", "{input_image}"]


# ---------------------------------------------------------------------- #
# SamRefinerAdapter — dry run
# ---------------------------------------------------------------------- #
def test_samrefiner_dry_run_resolves(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path)
    res, outputs = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert str(outputs["refined_mask"]).endswith("refined_mask.png")
    assert str(outputs["mask_quality"]).endswith("mask_quality.json")
    assert "img.png" in res.command
    assert "coarse.png" in res.command
    # Files not actually written in dry-run.
    assert not outputs["refined_mask"].exists()


# ---------------------------------------------------------------------- #
# SamRefinerAdapter — real run with mock command
# ---------------------------------------------------------------------- #
def test_samrefiner_run_mock_writes_outputs_and_validates(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path, command_template=_MOCK_REFINE)
    res, outputs = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=True)
    assert res.dry_run is False
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["refined_mask"].exists()
    assert outputs["mask_quality"].exists()
    assert json.loads(outputs["mask_quality"].read_text()) == {"boundary_f1": 0.9}


def test_samrefiner_run_default_template_fails_in_cpu_env(tmp_path: Path):
    # The default template calls `python -m samrefiner.refine` which is not
    # installed in the CPU test env -> non-zero returncode, missing outputs.
    adapter = SamRefinerAdapter(tmp_path)
    res, _ = adapter.refine("img.png", "coarse.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False
    assert res.missing_outputs == ["refined_mask", "mask_quality"]


# ---------------------------------------------------------------------- #
# SamRefinerAdapter — refine_batch + provenance JSONL
# ---------------------------------------------------------------------- #
def test_samrefiner_refine_batch_writes_provenance(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path, command_template=_MOCK_REFINE)
    items = [
        {"item_id": "i1", "instance_id": "p1", "image": "a.png", "coarse_mask": "c1.png"},
        {"item_id": "i2", "instance_id": "p2", "image": "b.png", "coarse_mask": "c2.png"},
    ]
    prov_path = tmp_path / "provenance.jsonl"
    rows = adapter.refine_batch(items, output_root=tmp_path / "refined", provenance_path=prov_path)
    assert len(rows) == 2
    assert rows[0]["item_id"] == "i1"
    assert rows[0]["ok"] is True
    assert rows[0]["branch_source"]["adapter"] == "samrefiner"
    assert rows[0]["branch_source"]["stage"] == "mask_refine"
    assert rows[0]["branch_source"]["item_id"] == "i1"
    assert "license_review" in rows[0]["license_meta"]
    # Per-item output dirs created.
    assert (tmp_path / "refined" / "i1" / "refined_mask.png").exists()
    assert (tmp_path / "refined" / "i2" / "mask_quality.json").exists()
    # Provenance JSONL written.
    lines = prov_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["item_id"] == "i1"
    assert row0["ok"] is True


def test_samrefiner_refine_batch_dry_run_no_outputs(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path)
    items = [{"item_id": "i1", "instance_id": "p1", "image": "a.png", "coarse_mask": "c.png"}]
    rows = adapter.refine_batch(items, output_root=tmp_path / "refined", execute=False)
    assert rows[0]["dry_run"] is True
    assert rows[0]["ok"] is True
    # Dry run doesn't write output files.
    assert not (tmp_path / "refined" / "i1" / "refined_mask.png").exists()


def test_samrefiner_refine_batch_default_ids_for_missing(tmp_path: Path):
    adapter = SamRefinerAdapter(tmp_path, command_template=_MOCK_REFINE)
    items = [{"image": "a.png", "coarse_mask": "c.png"}]  # no ids
    rows = adapter.refine_batch(items, output_root=tmp_path / "r")
    assert rows[0]["item_id"] == "item0"
    assert rows[0]["instance_id"] == "inst0"


# ---------------------------------------------------------------------- #
# HqSamAdapter
# ---------------------------------------------------------------------- #
def test_hq_sam_adapter_uses_registry_spec(tmp_path: Path):
    adapter = HqSamAdapter(tmp_path)
    assert adapter.spec.name == "hq_sam"
    assert adapter.spec.expected_outputs == ["refined_mask"]


def test_hq_sam_dry_run_resolves_box(tmp_path: Path):
    adapter = HqSamAdapter(tmp_path)
    res, outputs = adapter.refine("img.png", "10,10,40,50", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "10,10,40,50" in res.command
    assert "img.png" in res.command
    assert str(outputs["refined_mask"]).endswith("refined_mask.png")


def test_hq_sam_run_mock_writes_output(tmp_path: Path):
    adapter = HqSamAdapter(tmp_path, command_template=_MOCK_HQ)
    res, outputs = adapter.refine("img.png", "10,10,40,50", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["refined_mask"].exists()


def test_hq_sam_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = HqSamAdapter(tmp_path)
    res, _ = adapter.refine("img.png", "10,10,40,50", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert res.ok is False