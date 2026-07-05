"""Tests for hmp.config (Step 00)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hmp.config import Config, load_config, resolve_path, seed_from_config


def test_config_attribute_access_nested():
    cfg = Config({"paths": {"raw_dir": "data/raw"}, "project": {"seed": 7}})
    assert cfg.paths.raw_dir == "data/raw"
    assert cfg.project.seed == 7


def test_config_missing_key_raises_attribute_error():
    cfg = Config({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.does_not_exist


def test_config_get_default():
    cfg = Config({"a": 1})
    assert cfg.get("missing", "fallback") == "fallback"
    assert cfg.get("a") == 1


def test_config_contains_and_to_dict():
    cfg = Config({"a": {"b": 2}})
    assert "a" in cfg
    assert "missing" not in cfg
    assert cfg.to_dict() == {"a": {"b": 2}}


def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("project:\n  name: x\n  seed: 42\npaths:\n  raw_dir: data/raw\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.project.name == "x"
    assert cfg.paths.raw_dir == "data/raw"
    assert seed_from_config(cfg) == 42


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping_root(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_resolve_path_relative_and_absolute(tmp_path):
    base = tmp_path / "proj"
    base.mkdir()
    assert resolve_path(base, "data/raw") == base / "data" / "raw"
    abs_path = "/tmp/some/abs"
    assert resolve_path(base, abs_path) == Path(abs_path)


def test_seed_from_config_default():
    cfg = Config({"project": {}})
    assert seed_from_config(cfg) == 42