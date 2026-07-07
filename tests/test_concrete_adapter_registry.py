"""Tests for the concrete-adapter registry (CONCRETE_ADAPTERS / get_concrete_adapter_class)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hmp.adapters import (
    CONCRETE_ADAPTERS,
    get_concrete_adapter_class,
    load_registry,
)
from hmp.adapters.base import SubprocessAdapter
from hmp.cli import app

runner = CliRunner()


# The set of integrations that have a concrete typed adapter. This MUST stay
# in sync with the drift-prevention `concrete` list in test_adapters_templates.py
# and with the CONCRETE_ADAPTERS map in src/hmp/adapters/__init__.py.
EXPECTED_CONCRETE = {
    "samrefiner", "hq_sam", "cascadepsp",
    "matanyone", "matanyone2", "semat", "matting_anything", "maggie", "rvm",
    "cutie", "xmem",
    "videomama", "diffmatte", "sdmatte",
    "raft", "mmagic",
    "grounded_sam2", "groundingdino", "ultralytics_yolo",
    "fiftyone", "cvat", "label_studio",
    "gymnasium", "stable_baselines3",
}


def test_concrete_registry_lazy_and_complete():
    cls = get_concrete_adapter_class("samrefiner")
    assert cls is not None
    assert set(CONCRETE_ADAPTERS.keys()) == EXPECTED_CONCRETE


def test_every_concrete_class_is_subprocess_adapter_subclass():
    assert CONCRETE_ADAPTERS is not None
    for name, cls in CONCRETE_ADAPTERS.items():
        assert issubclass(cls, SubprocessAdapter), f"{name}: {cls} not SubprocessAdapter"


def test_concrete_registry_covers_only_existing_registry_integrations():
    reg_names = set(load_registry().names())
    assert set(CONCRETE_ADAPTERS.keys()).issubset(reg_names)
    # sam2 is the one registry integration without a concrete adapter (A2/GPU).
    assert "sam2" in reg_names
    assert get_concrete_adapter_class("sam2") is None


def test_get_concrete_adapter_class_unknown_returns_none():
    assert get_concrete_adapter_class("does_not_exist") is None


def test_concrete_class_constructs_from_registry_spec(tmp_path: Path):
    for name in EXPECTED_CONCRETE:
        cls = get_concrete_adapter_class(name)
        adapter = cls(tmp_path)
        assert adapter.spec.name == name
        # output keys in the spec are exactly what validate_outputs will check
        reg = load_registry()
        assert set(adapter.spec.expected_outputs) == set(reg.get(name).expected_outputs)


# ---------------------------------------------------------------------- #
# CLI `adapters list --typed`
# ---------------------------------------------------------------------- #
def test_adapters_list_typed_filter_only_concrete():
    result = runner.invoke(app, ["adapters", "list", "--typed"])
    assert result.exit_code == 0
    lines = [l for l in result.output.strip().splitlines() if l]
    names = {l.split("\t")[0] for l in lines}
    assert names == EXPECTED_CONCRETE
    # every line's typed column (index 5) is "typed"
    for l in lines:
        assert l.split("\t")[4] == "typed"


def test_adapters_list_shows_typed_column_for_concrete_and_generic():
    result = runner.invoke(app, ["adapters", "list"])
    assert result.exit_code == 0
    lines = {l.split("\t")[0]: l for l in result.output.strip().splitlines() if l}
    assert lines["samrefiner"].split("\t")[4] == "typed"
    assert lines["sam2"].split("\t")[4] == "generic"


def test_adapters_list_typed_filter_combined_with_group():
    # cvat + label_studio are in the hitl group; fiftyone is data_management.
    result = runner.invoke(app, ["adapters", "list", "--group", "hitl", "--typed"])
    assert result.exit_code == 0
    names = {l.split("\t")[0] for l in result.output.strip().splitlines() if l}
    assert names == {"cvat", "label_studio"}


def test_adapters_list_typed_filter_data_management_group():
    result = runner.invoke(app, ["adapters", "list", "--group", "data_management", "--typed"])
    assert result.exit_code == 0
    names = {l.split("\t")[0] for l in result.output.strip().splitlines() if l}
    assert names == {"fiftyone"}