"""Tests for FiftyoneAdapter, CvatAdapter, LabelStudioAdapter, GymnasiumAdapter, StableBaselines3Adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from hmp.adapters.active_labeling import GymnasiumAdapter, StableBaselines3Adapter
from hmp.adapters.hitl import CvatAdapter, FiftyoneAdapter, LabelStudioAdapter


def _mock(*out_files: str) -> list[str]:
    parts = ["import json,pathlib"]
    for f in out_files:
        if f.endswith(".json"):
            parts.append(f"pathlib.Path('{f}').write_text(json.dumps({{'v':0.9}}))")
        elif f.endswith(".zip"):
            parts.append(f"pathlib.Path('{f}').write_bytes(b'PK')")
        else:
            parts.append(f"pathlib.Path('{f}').mkdir(parents=True,exist_ok=True)")
    return [sys.executable, "-c", "; ".join(parts)]


# ---------------------------------------------------------------------- #
# FiftyoneAdapter
# ---------------------------------------------------------------------- #
def test_fiftyone_uses_registry_spec(tmp_path: Path):
    adapter = FiftyoneAdapter(tmp_path)
    assert adapter.spec.name == "fiftyone"
    assert set(adapter.spec.expected_outputs) == {"dataset_view", "review_selection"}


def test_fiftyone_dry_run_resolves(tmp_path: Path):
    adapter = FiftyoneAdapter(tmp_path, view_spec="high_uncertainty")
    res, outputs = adapter.review("data/", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "data/" in res.command
    assert "high_uncertainty" in res.command
    assert "--dataset-view" in res.command
    assert str(outputs["dataset_view"]).endswith("dataset_view.json")
    assert str(outputs["review_selection"]).endswith("review_selection.json")


def test_fiftyone_run_mock_creates_all(tmp_path: Path):
    adapter = FiftyoneAdapter(
        tmp_path, command_template=_mock("{output_dataset_view}", "{output_review_selection}")
    )
    res, outputs = adapter.review("data/", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["dataset_view"].exists() and outputs["review_selection"].exists()


def test_fiftyone_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = FiftyoneAdapter(tmp_path)
    res, _ = adapter.review("data/", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"dataset_view", "review_selection"}


# ---------------------------------------------------------------------- #
# CvatAdapter (no input files, only task_id param)
# ---------------------------------------------------------------------- #
def test_cvat_uses_registry_spec(tmp_path: Path):
    adapter = CvatAdapter(tmp_path)
    assert adapter.spec.name == "cvat"
    assert set(adapter.spec.expected_outputs) == {"human_edits", "corrected_prompts", "audit_log"}


def test_cvat_dry_run_resolves(tmp_path: Path):
    adapter = CvatAdapter(tmp_path, task_id="task-42")
    res, outputs = adapter.correct(output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "task-42" in res.command
    assert "--corrected-prompts" in res.command
    assert str(outputs["corrected_prompts"]).endswith("corrected_prompts.json")


def test_cvat_run_mock_creates_all(tmp_path: Path):
    adapter = CvatAdapter(
        tmp_path, task_id="t1",
        command_template=_mock("{output_human_edits}", "{output_corrected_prompts}", "{output_audit_log}"),
    )
    res, outputs = adapter.correct(output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    for p in outputs.values():
        assert p.exists()


def test_cvat_run_partial_reports_missing(tmp_path: Path):
    adapter = CvatAdapter(
        tmp_path, task_id="t1", command_template=_mock("{output_human_edits}", "{output_audit_log}")
    )
    res, _ = adapter.correct(output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.missing_outputs == ["corrected_prompts"]
    assert res.ok is False


def test_cvat_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = CvatAdapter(tmp_path, task_id="t1")
    res, _ = adapter.correct(output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"human_edits", "corrected_prompts", "audit_log"}


# ---------------------------------------------------------------------- #
# LabelStudioAdapter
# ---------------------------------------------------------------------- #
def test_label_studio_uses_registry_spec(tmp_path: Path):
    adapter = LabelStudioAdapter(tmp_path)
    assert adapter.spec.name == "label_studio"
    assert set(adapter.spec.expected_outputs) == {"human_edits", "audit_log"}


def test_label_studio_dry_run_resolves(tmp_path: Path):
    adapter = LabelStudioAdapter(tmp_path, project_id="proj-7")
    res, outputs = adapter.correct(output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "proj-7" in res.command
    assert str(outputs["audit_log"]).endswith("audit_log.json")


def test_label_studio_run_mock_creates_all(tmp_path: Path):
    adapter = LabelStudioAdapter(
        tmp_path, project_id="p1",
        command_template=_mock("{output_human_edits}", "{output_audit_log}"),
    )
    res, outputs = adapter.correct(output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["human_edits"].exists() and outputs["audit_log"].exists()


def test_label_studio_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = LabelStudioAdapter(tmp_path, project_id="p1")
    res, _ = adapter.correct(output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"human_edits", "audit_log"}


# ---------------------------------------------------------------------- #
# GymnasiumAdapter
# ---------------------------------------------------------------------- #
def test_gymnasium_uses_registry_spec(tmp_path: Path):
    adapter = GymnasiumAdapter(tmp_path)
    assert adapter.spec.name == "gymnasium"
    assert set(adapter.spec.expected_outputs) == {"agent_episode", "reward_trace"}


def test_gymnasium_dry_run_resolves(tmp_path: Path):
    adapter = GymnasiumAdapter(tmp_path)
    res, outputs = adapter.rollout("env.yaml", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "env.yaml" in res.command
    assert "--episode-out" in res.command
    assert str(outputs["agent_episode"]).endswith("agent_episode.json")


def test_gymnasium_run_mock_creates_all(tmp_path: Path):
    adapter = GymnasiumAdapter(
        tmp_path, command_template=_mock("{output_agent_episode}", "{output_reward_trace}")
    )
    res, outputs = adapter.rollout("env.yaml", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["agent_episode"].exists() and outputs["reward_trace"].exists()


def test_gymnasium_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = GymnasiumAdapter(tmp_path)
    res, _ = adapter.rollout("env.yaml", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"agent_episode", "reward_trace"}


# ---------------------------------------------------------------------- #
# StableBaselines3Adapter
# ---------------------------------------------------------------------- #
def test_sb3_uses_registry_spec(tmp_path: Path):
    adapter = StableBaselines3Adapter(tmp_path)
    assert adapter.spec.name == "stable_baselines3"
    assert set(adapter.spec.expected_outputs) == {"policy_checkpoint", "decision_trace"}


def test_sb3_dry_run_resolves(tmp_path: Path):
    adapter = StableBaselines3Adapter(tmp_path)
    res, outputs = adapter.train("env.yaml", output_dir=tmp_path / "out", execute=False)
    assert res.dry_run is True
    assert res.ok is True
    assert "env.yaml" in res.command
    assert "--checkpoint" in res.command
    assert str(outputs["policy_checkpoint"]).endswith("policy_checkpoint.zip")
    assert str(outputs["decision_trace"]).endswith("decision_trace.json")


def test_sb3_run_mock_creates_all(tmp_path: Path):
    adapter = StableBaselines3Adapter(
        tmp_path, command_template=_mock("{output_policy_checkpoint}", "{output_decision_trace}")
    )
    res, outputs = adapter.train("env.yaml", output_dir=tmp_path / "out", execute=True)
    assert res.returncode == 0
    assert res.ok is True
    assert outputs["policy_checkpoint"].exists() and outputs["decision_trace"].exists()


def test_sb3_run_default_template_fails_in_cpu_env(tmp_path: Path):
    adapter = StableBaselines3Adapter(tmp_path)
    res, _ = adapter.train("env.yaml", output_dir=tmp_path / "out", execute=True)
    assert res.returncode != 0
    assert set(res.missing_outputs) == {"policy_checkpoint", "decision_trace"}


# ---------------------------------------------------------------------- #
# Registry drift sanity
# ---------------------------------------------------------------------- #
def test_hitl_and_rl_keys_match_registry(tmp_path: Path):
    from hmp.adapters import load_registry

    reg = load_registry()
    assert set(FiftyoneAdapter(tmp_path).spec.expected_outputs) == set(reg.get("fiftyone").expected_outputs)
    assert set(CvatAdapter(tmp_path).spec.expected_outputs) == set(reg.get("cvat").expected_outputs)
    assert set(LabelStudioAdapter(tmp_path).spec.expected_outputs) == set(reg.get("label_studio").expected_outputs)
    assert set(GymnasiumAdapter(tmp_path).spec.expected_outputs) == set(reg.get("gymnasium").expected_outputs)
    assert set(StableBaselines3Adapter(tmp_path).spec.expected_outputs) == set(reg.get("stable_baselines3").expected_outputs)