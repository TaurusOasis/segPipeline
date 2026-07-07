"""Tests for the adapter command-template catalog + build_adapter factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hmp.adapters import (
    DEFAULT_COMMAND_TEMPLATES,
    ADAPTER_INPUT_KEYS,
    ADAPTER_OUTPUT_KEYS,
    build_adapter,
    dry_run_adapter,
    load_registry,
    template_for,
)
from hmp.adapters import SubprocessAdapter
from hmp.adapters.base import MissingPlaceholder


# ---------------------------------------------------------------------- #
# Catalog coverage vs registry
# ---------------------------------------------------------------------- #
def test_every_registry_integration_has_a_template():
    reg = load_registry()
    missing = sorted(set(reg.names()) - set(DEFAULT_COMMAND_TEMPLATES))
    assert missing == [], f"integrations without a command template: {missing}"


def test_template_for_returns_copy():
    t1 = template_for("samrefiner")
    t2 = template_for("samrefiner")
    assert t1 == t2
    t1.append("--mutated")
    assert "--mutated" not in template_for("samrefiner")


def test_template_for_unknown_raises():
    with pytest.raises(KeyError):
        template_for("does_not_exist")


def test_input_output_keys_subset_of_template_placeholders():
    # Every documented input/output key must appear as a placeholder in the
    # corresponding template, else dry-run could never resolve it.
    for name, keys in ADAPTER_INPUT_KEYS.items():
        tmpl = " ".join(DEFAULT_COMMAND_TEMPLATES[name])
        for k in keys:
            assert f"{{input_{k}}}" in tmpl, f"{name}: input {k!r} not in template"
    for name, keys in ADAPTER_OUTPUT_KEYS.items():
        tmpl = " ".join(DEFAULT_COMMAND_TEMPLATES[name])
        for k in keys:
            assert f"{{output_{k}}}" in tmpl, f"{name}: output {k!r} not in template"


def test_concrete_adapter_output_keys_match_registry_expected_outputs():
    # Integrations that have a concrete typed adapter (samrefiner, hq_sam,
    # raft, gmflow, matanyone) must produce outputs whose keys exactly match
    # the registry expected_outputs, so validate_outputs passes.
    reg = load_registry()
    concrete = ["samrefiner", "hq_sam", "raft", "gmflow", "matanyone"]
    for name in concrete:
        assert set(ADAPTER_OUTPUT_KEYS[name]) == set(reg.get(name).expected_outputs), (
            f"{name}: ADAPTER_OUTPUT_KEYS {ADAPTER_OUTPUT_KEYS[name]} != "
            f"registry expected_outputs {reg.get(name).expected_outputs}"
        )


# ---------------------------------------------------------------------- #
# build_adapter
# ---------------------------------------------------------------------- #
def test_build_adapter_assembles_subprocess_adapter(tmp_path: Path):
    adapter = build_adapter("samrefiner", workdir=tmp_path)
    assert isinstance(adapter, SubprocessAdapter)
    assert adapter.spec.name == "samrefiner"
    assert adapter.spec.expected_outputs == ["refined_mask", "mask_quality"]
    assert adapter.command_template == template_for("samrefiner")
    assert adapter.env_overlay["REPO_PYTHON"] == "python"


def test_build_adapter_default_workdir_is_per_name(tmp_path: Path, monkeypatch):
    # Default workdir is runs/adapters/<name> under cwd. Build the registry
    # from an absolute path so this test does not depend on base.py's
    # cwd-relative default registry path.
    from hmp.adapters.base import AdapterRegistry

    repo_root = Path(__file__).resolve().parents[1]
    reg = AdapterRegistry.from_yaml(repo_root / "configs" / "reference_integrations.yaml")
    monkeypatch.chdir(tmp_path)
    adapter = build_adapter("sam2", registry=reg)
    assert adapter.workdir.name == "sam2"
    assert adapter.workdir.parent.name == "adapters"


def test_build_adapter_command_template_override(tmp_path: Path):
    custom = ["echo", "{input_image}"]
    adapter = build_adapter("hq_sam", workdir=tmp_path, command_template=custom)
    assert adapter.command_template == ["echo", "{input_image}"]


def test_build_adapter_env_overlay_keeps_explicit_values(tmp_path: Path):
    adapter = build_adapter("raft", workdir=tmp_path, env={"REPO_PYTHON": "/opt/venv/bin/python", "CUDA_VISIBLE_DEVICES": "0"})
    assert adapter.env_overlay["REPO_PYTHON"] == "/opt/venv/bin/python"
    assert adapter.env_overlay["CUDA_VISIBLE_DEVICES"] == "0"


def test_build_adapter_unknown_integration_raises(tmp_path: Path):
    with pytest.raises(KeyError):
        build_adapter("nope", workdir=tmp_path)


# ---------------------------------------------------------------------- #
# dry_run_adapter
# ---------------------------------------------------------------------- #
def test_dry_run_adapter_resolves_samrefiner(tmp_path: Path):
    res = dry_run_adapter(
        "samrefiner",
        workdir=tmp_path,
        inputs={"image": "/data/a.png", "coarse_mask": "/data/coarse.png"},
        outputs={"refined_mask": tmp_path / "ref.png", "mask_quality": tmp_path / "q.json"},
        params={"repo_python": sys.executable},
    )
    assert res.dry_run is True
    assert res.ok is True
    assert res.command[0] == sys.executable
    assert "--image" in res.command
    assert "/data/a.png" in res.command
    assert str(tmp_path / "ref.png") in res.command
    assert res.outputs["refined_mask"] == str(tmp_path / "ref.png")


def test_dry_run_adapter_supplies_default_repo_python(tmp_path: Path):
    res = dry_run_adapter(
        "samrefiner",
        workdir=tmp_path,
        inputs={"image": "a.png", "coarse_mask": "c.png"},
        outputs={"refined_mask": tmp_path / "r.png", "mask_quality": tmp_path / "q.json"},
    )
    # default repo_python is "python" when not supplied.
    assert res.command[0] == "python"
    assert res.env["REPO_PYTHON"] == "python"


def test_dry_run_adapter_missing_input_raises(tmp_path: Path):
    with pytest.raises(MissingPlaceholder):
        dry_run_adapter(
            "samrefiner",
            workdir=tmp_path,
            inputs={"image": "a.png"},  # coarse_mask missing
            outputs={"refined_mask": tmp_path / "r.png", "mask_quality": tmp_path / "q.json"},
        )


def test_dry_run_adapter_sam2_uses_dir_inputs(tmp_path: Path):
    res = dry_run_adapter(
        "sam2",
        workdir=tmp_path,
        inputs={"image_dir": "/frames", "keyframe_box": "10,10,40,50"},
        outputs={"masklet_dir": tmp_path / "masklet", "track_id": tmp_path / "track.json"},
    )
    assert "--image-dir" in res.command
    assert "/frames" in res.command
    assert "10,10,40,50" in res.command
    assert res.outputs["track_id"] == str(tmp_path / "track.json")


# ---------------------------------------------------------------------- #
# validate_outputs uses spec.expected_outputs
# ---------------------------------------------------------------------- #
def test_validate_outputs_for_built_adapter(tmp_path: Path):
    adapter = build_adapter("samrefiner", workdir=tmp_path)
    # Only refined_mask exists; mask_quality is missing.
    ref = tmp_path / "ref.png"
    ref.write_text("x")
    missing = adapter.validate_outputs(
        {"refined_mask": ref, "mask_quality": tmp_path / "q.json"}
    )
    assert missing == ["mask_quality"]


# ---------------------------------------------------------------------- #
# provenance from a built adapter
# ---------------------------------------------------------------------- #
def test_provenance_from_built_adapter(tmp_path: Path):
    adapter = build_adapter("samrefiner", workdir=tmp_path)
    res = adapter.dry_run(
        inputs={"image": "a.png", "coarse_mask": "c.png"},
        outputs={"refined_mask": tmp_path / "r.png", "mask_quality": tmp_path / "q.json"},
        params={"repo_python": "python"},
    )
    prov = adapter.provenance(res, branch_source={"stage": "mask_refine"})
    assert prov["branch_source"] == {"adapter": "samrefiner", "group": "mask_refine", "stage": "mask_refine"}
    assert prov["license_meta"]["url"].startswith("https://github.com/")
    assert prov["prompt_history"][0]["command"][0] == "python"


# ---------------------------------------------------------------------- #
# Real run with a no-op command (proves the factory produces a runnable adapter)
# ---------------------------------------------------------------------- #
def test_build_adapter_run_executes(tmp_path: Path):
    adapter = build_adapter(
        "samrefiner",
        workdir=tmp_path,
        command_template=[sys.executable, "-c", "print('ok')"],
    )
    res = adapter.run(inputs={}, outputs={})
    assert res.returncode == 0
    assert "ok" in res.stdout
    # No expected outputs produced -> ok False, missing_outputs lists them.
    assert res.missing_outputs == ["refined_mask", "mask_quality"]
    assert res.ok is False