"""Tests for the RaftAdapter (QA, CPU fallback) and MatAnyoneAdapter (Bv)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hmp.adapters.matting import MatAnyoneAdapter
from hmp.adapters.qa import RaftAdapter


# ---------------------------------------------------------------------- #
# RaftAdapter — construction
# ---------------------------------------------------------------------- #
def test_raft_adapter_uses_registry_spec(tmp_path: Path):
    adapter = RaftAdapter(tmp_path)
    assert adapter.spec.name == "raft"
    assert set(adapter.spec.expected_outputs) == {"temporal_error", "flow_consistency_score"}
    assert adapter.fallback is True
    assert "raft.compute_flow" in " ".join(adapter.command_template)


def test_raft_adapter_command_template_override(tmp_path: Path):
    adapter = RaftAdapter(tmp_path, command_template=["echo", "{input_prev_alpha}"])
    assert adapter.command_template == ["echo", "{input_prev_alpha}"]


# ---------------------------------------------------------------------- #
# RaftAdapter — dry run
# ---------------------------------------------------------------------- #
def test_raft_dry_run_resolves(tmp_path: Path):
    adapter = RaftAdapter(tmp_path)
    res, outputs = adapter.run_qa("prev.png", "cur.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "prev.png" in res.command and "cur.png" in res.command
    assert str(outputs["temporal_error"]).endswith("temporal_error.json")
    assert str(outputs["flow_consistency_score"]).endswith("flow_consistency_score.json")
    assert not outputs["temporal_error"].exists()


# ---------------------------------------------------------------------- #
# RaftAdapter — CPU fallback (default template, no RAFT repo)
# ---------------------------------------------------------------------- #
def test_raft_run_cpu_fallback_writes_outputs(tmp_path: Path):
    a = np.zeros((8, 8), dtype=np.float32)
    a[2:6, 2:6] = 1.0
    b = np.zeros((8, 8), dtype=np.float32)
    b[3:7, 3:7] = 1.0  # shifted block -> non-zero flicker
    adapter = RaftAdapter(tmp_path)  # default template -> fails in CPU env
    res, outputs = adapter.run_qa(a, b, output_dir=tmp_path / "out", execute=True)
    # Fallback recovered the run.
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert "cpu fallback" in res.stderr
    err = json.loads(outputs["temporal_error"].read_text())
    assert "temporal_error" in err and err["temporal_error"] >= 0.0
    cons = json.loads(outputs["flow_consistency_score"].read_text())
    # Block shifted -> some flicker -> consistency < 1.
    assert 0.0 <= cons["flow_consistency_score"] < 1.0


def test_raft_run_cpu_fallback_identical_frames_full_consistency(tmp_path: Path):
    a = np.full((8, 8), 0.5, dtype=np.float32)
    adapter = RaftAdapter(tmp_path)
    res, outputs = adapter.run_qa(a, a, output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    cons = json.loads(outputs["flow_consistency_score"].read_text())
    assert cons["flow_consistency_score"] == pytest.approx(1.0)


def test_raft_run_fallback_disabled_reports_failure(tmp_path: Path):
    a = np.zeros((8, 8), dtype=np.float32)
    adapter = RaftAdapter(tmp_path, fallback=False)
    res, outputs = adapter.run_qa(a, a, output_dir=tmp_path / "out", execute=True)
    # No fallback -> default template fails, outputs missing.
    assert res.returncode != 0
    assert res.ok is False
    assert res.missing_outputs == ["temporal_error", "flow_consistency_score"]
    assert not outputs["temporal_error"].exists()


def test_raft_run_mock_command_skips_fallback(tmp_path: Path):
    # A mock command that writes both outputs succeeds -> no fallback path.
    mock = [
        sys.executable, "-c",
        "import json,pathlib; "
        "pathlib.Path('{output_temporal_error}').write_text(json.dumps({{'temporal_error':0.1}})); "
        "pathlib.Path('{output_flow_consistency_score}').write_text(json.dumps({{'flow_consistency_score':0.9}}))",
    ]
    adapter = RaftAdapter(tmp_path, command_template=mock)
    res, outputs = adapter.run_qa("prev.png", "cur.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert "cpu fallback" not in (res.stderr or "")
    assert json.loads(outputs["temporal_error"].read_text())["temporal_error"] == pytest.approx(0.1)


# ---------------------------------------------------------------------- #
# RaftAdapter — fallback loads alpha from path
# ---------------------------------------------------------------------- #
def test_raft_cpu_fallback_loads_alpha_from_path(tmp_path: Path):
    from PIL import Image

    prev_path = tmp_path / "prev.png"
    cur_path = tmp_path / "cur.png"
    Image.fromarray(np.full((8, 8), 200, dtype=np.uint8)).save(prev_path)
    Image.fromarray(np.full((8, 8), 50, dtype=np.uint8)).save(cur_path)
    adapter = RaftAdapter(tmp_path)
    res, outputs = adapter.run_qa(prev_path, cur_path, output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    err = json.loads(outputs["temporal_error"].read_text())
    # 200/255 vs 50/255 -> |0.784 - 0.196| ~ 0.588 flicker.
    assert err["flicker"] == pytest.approx(0.588, abs=0.01)


# ---------------------------------------------------------------------- #
# MatAnyoneAdapter — construction
# ---------------------------------------------------------------------- #
def test_matanyone_adapter_uses_registry_spec(tmp_path: Path):
    adapter = MatAnyoneAdapter(tmp_path)
    assert adapter.spec.name == "matanyone"
    assert set(adapter.spec.expected_outputs) == {"alpha_video", "branch_source"}
    assert "matanyone.infer" in " ".join(adapter.command_template)


# ---------------------------------------------------------------------- #
# MatAnyoneAdapter — dry run
# ---------------------------------------------------------------------- #
def test_matanyone_dry_run_resolves(tmp_path: Path):
    adapter = MatAnyoneAdapter(tmp_path)
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "/frames" in res.command and "target.png" in res.command
    assert str(outputs["alpha_video"]).endswith("alpha.mp4")
    assert str(outputs["branch_source"]).endswith("branch_source.json")
    # Dry run does not write branch_source stub.
    assert not outputs["branch_source"].exists()


# ---------------------------------------------------------------------- #
# MatAnyoneAdapter — real run, default template fails in CPU env
# ---------------------------------------------------------------------- #
def test_matanyone_run_default_template_writes_branch_source_stub(tmp_path: Path):
    adapter = MatAnyoneAdapter(tmp_path)
    res, outputs = adapter.mat(
        "/frames", "target.png", output_dir=tmp_path / "out", execute=True, target_id="t01"
    )
    # alpha_video missing (repo not present), branch_source stub written.
    assert res.returncode != 0
    assert res.ok is False
    assert res.missing_outputs == ["alpha_video"]
    assert outputs["branch_source"].exists()
    stub = json.loads(outputs["branch_source"].read_text())
    assert stub["adapter"] == "matanyone"
    assert stub["branch"] == "Bv"
    assert stub["target_id"] == "t01"


# ---------------------------------------------------------------------- #
# MatAnyoneAdapter — mock command writes alpha + branch_source
# ---------------------------------------------------------------------- #
def test_matanyone_run_mock_command_succeeds(tmp_path: Path):
    mock = [
        sys.executable, "-c",
        "import json,pathlib; "
        "pathlib.Path('{output_alpha_video}').write_bytes(b'\\x89\\x10\\x00\\x10'); "
        "pathlib.Path('{output_branch_source}').write_text(json.dumps({{'branch':'Bv'}}))",
    ]
    adapter = MatAnyoneAdapter(tmp_path, command_template=mock)
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert outputs["alpha_video"].exists()
    # External CLI wrote branch_source -> stub not overwritten.
    assert json.loads(outputs["branch_source"].read_text()) == {"branch": "Bv"}


def test_matanyone_run_mock_no_branch_source_still_stubs(tmp_path: Path):
    # Mock writes only alpha_video -> branch_source stub written by adapter.
    mock = [
        sys.executable, "-c",
        "import pathlib; pathlib.Path('{output_alpha_video}').write_bytes(b'\\x89\\x10\\x00\\x10')",
    ]
    adapter = MatAnyoneAdapter(tmp_path, command_template=mock)
    res, outputs = adapter.mat("/frames", "target.png", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert json.loads(outputs["branch_source"].read_text())["adapter"] == "matanyone"