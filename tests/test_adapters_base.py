"""Tests for the external adapter base contract (src/hmp/adapters)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from hmp.adapters import (
    AdapterRegistry,
    AdapterResult,
    AdapterSpec,
    ExternalAdapter,
    SubprocessAdapter,
    load_registry,
)
from hmp.adapters.base import MissingPlaceholder

# ExternalAdapter is a concrete base (full default implementation); the
# generic SubprocessAdapter is a thin named subclass. Both are directly
# instantiable.


# ---------------------------------------------------------------------- #
# AdapterSpec
# ---------------------------------------------------------------------- #
def test_adapter_spec_from_registry_entry():
    spec = AdapterSpec.from_registry_entry(
        "sam2",
        {
            "group": "masklet",
            "url": "https://github.com/facebookresearch/sam2",
            "adapter_target": "src/hmp/adapters/vos/sam2.py",
            "expected_outputs": ["masklet", "track_id", "prompt_history"],
            "role": "Promptable image/video segmentation.",
            "priority": 1,
            "license_review": "required",
        },
    )
    assert spec.name == "sam2"
    assert spec.group == "masklet"
    assert spec.expected_outputs == ["masklet", "track_id", "prompt_history"]
    assert spec.priority == 1
    assert spec.license_review == "required"


def test_adapter_spec_defaults():
    spec = AdapterSpec.from_registry_entry("x", {})
    assert spec.group == ""
    assert spec.expected_outputs == []
    assert spec.priority == 5
    assert spec.license_review == "required"


def test_adapter_spec_to_dict_roundtrip():
    spec = AdapterSpec(name="x", group="g", url="u", adapter_target="a")
    d = spec.to_dict()
    assert d["name"] == "x"
    assert d["group"] == "g"


# ---------------------------------------------------------------------- #
# Command building
# ---------------------------------------------------------------------- #
def _spec(name: str = "fake") -> AdapterSpec:
    return AdapterSpec(
        name=name,
        group="mask_refine",
        url="https://example/fake",
        adapter_target="src/hmp/adapters/mask_refine/fake.py",
        expected_outputs=["refined_mask"],
        role="fake",
        priority=2,
        license_review="required",
    )


def test_build_command_resolves_inputs_and_outputs(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[
            "python", "-c", "print('{input_image}', '{output_refined_mask}')",
        ],
    )
    cmd = adapter.build_command(
        inputs={"image": "/data/img.png"},
        outputs={"refined_mask": tmp_path / "out.png"},
    )
    assert cmd[2] == f"print('/data/img.png', '{tmp_path / 'out.png'}')"


def test_build_command_supports_params(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=["echo", "{input_image}", "{param_threshold}"],
    )
    cmd = adapter.build_command(
        inputs={"image": "a.png"},
        outputs={},
        params={"threshold": 0.5},
    )
    assert cmd == ["echo", "a.png", "0.5"]


def test_build_command_missing_placeholder_raises(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=["echo", "{input_image}", "{output_missing}"],
    )
    with pytest.raises(MissingPlaceholder):
        adapter.build_command(inputs={"image": "a.png"}, outputs={})


def test_build_command_no_template_raises(tmp_path: Path):
    adapter = SubprocessAdapter(_spec(), workdir=tmp_path)
    with pytest.raises(ValueError):
        adapter.build_command(inputs={}, outputs={})


def test_build_command_literal_braces_escape(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=["echo", "{{not a placeholder}}", "{input_image}"],
    )
    cmd = adapter.build_command(inputs={"image": "a.png"}, outputs={})
    assert cmd[1] == "{not a placeholder}"


# ---------------------------------------------------------------------- #
# Dry run
# ---------------------------------------------------------------------- #
def test_dry_run_does_not_execute(tmp_path: Path):
    sentinel = tmp_path / "touched.txt"
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[
            sys.executable, "-c",
            f"open(r'{sentinel}', 'w').write('x')",
        ],
    )
    res = adapter.dry_run(inputs={}, outputs={"refined_mask": tmp_path / "out.png"})
    assert res.dry_run is True
    assert res.returncode == 0
    assert res.ok is True
    assert sentinel.exists() is False  # nothing executed
    assert res.outputs == {"refined_mask": str(tmp_path / "out.png")}


def test_dry_run_resolves_env_overlay(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=["echo", "hi"],
        env={"FAKE_ADAPTER_KEY": "42"},
    )
    res = adapter.dry_run(inputs={}, outputs={})
    assert res.env["FAKE_ADAPTER_KEY"] == "42"
    # Real env still present.
    assert "PATH" in res.env


# ---------------------------------------------------------------------- #
# Real run
# ---------------------------------------------------------------------- #
def test_run_success_and_output_validation(tmp_path: Path):
    out_file = tmp_path / "out.txt"
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[
            sys.executable, "-c",
            f"open(r'{out_file}', 'w').write('ok')",
        ],
    )
    res = adapter.run(inputs={}, outputs={"refined_mask": out_file})
    assert res.dry_run is False
    assert res.returncode == 0
    assert res.missing_outputs == []
    assert res.ok is True
    assert out_file.exists()
    assert res.duration_s >= 0.0


def test_run_missing_output_recorded_not_raised(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[sys.executable, "-c", "print('done')"],
    )
    res = adapter.run(inputs={}, outputs={"refined_mask": tmp_path / "nope.png"})
    assert res.returncode == 0
    assert res.missing_outputs == ["refined_mask"]
    assert res.ok is False


def test_run_nonzero_returncode_recorded(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )
    res = adapter.run(inputs={}, outputs={})
    assert res.returncode == 3
    assert res.ok is False


def test_run_env_overlay_visible_to_subprocess(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[sys.executable, "-c", "import os; print(os.environ['FAKE_KEY'])"],
        env={"FAKE_KEY": "hello-env"},
    )
    res = adapter.run(inputs={}, outputs={})
    assert res.returncode == 0
    assert "hello-env" in res.stdout


def test_run_timeout_returns_failure_result(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=[sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=0.5,
    )
    res = adapter.run(inputs={}, outputs={})
    assert res.returncode == -1
    assert res.ok is False
    assert "timeout" in res.stderr


# ---------------------------------------------------------------------- #
# validate_outputs uses spec.expected_outputs
# ---------------------------------------------------------------------- #
def test_validate_outputs_uses_spec_expected_outputs(tmp_path: Path):
    spec = AdapterSpec(
        name="multi",
        group="g",
        url="u",
        adapter_target="a",
        expected_outputs=["alpha", "eval_map"],
    )
    adapter = SubprocessAdapter(spec, workdir=tmp_path, command_template=["echo"])
    # Only 'alpha' exists on disk; 'eval_map' is missing.
    alpha_path = tmp_path / "alpha.png"
    alpha_path.write_text("x")
    missing = adapter.validate_outputs(
        {"alpha": alpha_path, "eval_map": tmp_path / "eval.png", "extra": tmp_path / "extra.png"}
    )
    assert missing == ["eval_map"]


def test_validate_outputs_no_spec_expected_checks_all_provided(tmp_path: Path):
    spec = AdapterSpec(name="n", group="g", url="u", adapter_target="a")  # no expected_outputs
    adapter = SubprocessAdapter(spec, workdir=tmp_path, command_template=["echo"])
    a = tmp_path / "a.png"
    a.write_text("x")
    missing = adapter.validate_outputs({"a": a, "b": tmp_path / "b.png"})
    assert missing == ["b"]


# ---------------------------------------------------------------------- #
# provenance
# ---------------------------------------------------------------------- #
def test_provenance_emits_schema_fields(tmp_path: Path):
    adapter = SubprocessAdapter(
        _spec(),
        workdir=tmp_path,
        command_template=["echo", "{input_image}"],
    )
    res = adapter.dry_run(inputs={"image": "a.png"}, outputs={})
    prov = adapter.provenance(
        res,
        branch_source={"track_id": "t01"},
        quality_scores={"boundary_f1": 0.82},
        extra_license_meta={"license": "Apache-2.0"},
    )
    assert prov["branch_source"] == {
        "adapter": "fake",
        "group": "mask_refine",
        "track_id": "t01",
    }
    assert prov["prompt_history"][0]["command"] == ["echo", "a.png"]
    assert prov["prompt_history"][0]["dry_run"] is True
    assert prov["quality_scores"] == {"boundary_f1": 0.82}
    assert prov["license_meta"]["license_review"] == "required"
    assert prov["license_meta"]["url"] == "https://example/fake"
    assert prov["license_meta"]["license"] == "Apache-2.0"


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
def _write_registry(tmp_path: Path) -> Path:
    path = tmp_path / "reference_integrations.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "policy": {"require_dry_run": True, "integration_style": "adapter_or_subprocess"},
                "integrations": {
                    "sam2": {
                        "group": "masklet",
                        "url": "https://github.com/facebookresearch/sam2",
                        "adapter_target": "src/hmp/adapters/vos/sam2.py",
                        "expected_outputs": ["masklet", "track_id"],
                        "priority": 1,
                        "license_review": "required",
                    },
                    "samrefiner": {
                        "group": "mask_refine",
                        "url": "https://github.com/linyq2117/SAMRefiner",
                        "adapter_target": "src/hmp/adapters/mask_refine/samrefiner.py",
                        "expected_outputs": ["refined_mask", "mask_quality"],
                        "priority": 1,
                        "license_review": "required",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_registry_from_yaml_loads_specs(tmp_path: Path):
    reg = AdapterRegistry.from_yaml(_write_registry(tmp_path))
    assert set(reg.names()) == {"sam2", "samrefiner"}
    sam2 = reg.get("sam2")
    assert sam2.group == "masklet"
    assert sam2.expected_outputs == ["masklet", "track_id"]


def test_registry_get_unknown_raises(tmp_path: Path):
    reg = AdapterRegistry.from_yaml(_write_registry(tmp_path))
    with pytest.raises(KeyError):
        reg.get("nope")


def test_registry_by_group(tmp_path: Path):
    reg = AdapterRegistry.from_yaml(_write_registry(tmp_path))
    refine = reg.by_group("mask_refine")
    assert [s.name for s in refine] == ["samrefiner"]


def test_registry_build_subprocess_adapter(tmp_path: Path):
    reg = AdapterRegistry.from_yaml(_write_registry(tmp_path))
    adapter = reg.build(
        "samrefiner",
        workdir=tmp_path / "wd",
        command_template=["echo", "{input_image}", "{output_refined_mask}"],
        env={"SAMREFINER_HOME": "/opt/sr"},
    )
    assert isinstance(adapter, SubprocessAdapter)
    assert adapter.spec.name == "samrefiner"
    assert adapter.env_overlay == {"SAMREFINER_HOME": "/opt/sr"}
    cmd = adapter.build_command(
        inputs={"image": "a.png"},
        outputs={"refined_mask": tmp_path / "m.png"},
    )
    assert cmd == ["echo", "a.png", str(tmp_path / "m.png")]


def test_load_registry_real_file():
    # The repo's real configs/reference_integrations.yaml must parse and
    # contain the canonical integrations the plan depends on.
    reg = load_registry()
    for name in ["sam2", "samrefiner", "hq_sam", "matanyone", "raft", "cvat"]:
        assert name in reg.names(), f"missing integration {name!r} in registry"
    sam2 = reg.get("sam2")
    assert "masklet" in sam2.expected_outputs
    assert sam2.adapter_target.endswith("sam2.py")


# ---------------------------------------------------------------------- #
# ExternalAdapter is a concrete base
# ---------------------------------------------------------------------- #
def test_external_adapter_is_directly_usable(tmp_path: Path):
    # The base class has a full default implementation, so it can be used
    # directly without subclassing.
    adapter = ExternalAdapter(_spec(), workdir=tmp_path, command_template=["echo", "{input_image}"])
    cmd = adapter.build_command(inputs={"image": "a.png"}, outputs={})
    assert cmd == ["echo", "a.png"]