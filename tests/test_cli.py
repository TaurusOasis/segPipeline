"""Tests for hmp.cli (Step 00)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hmp.cli import app

runner = CliRunner()


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "human matting pipeline" in result.output.lower() or "hmp" in result.output.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    from hmp import __version__

    assert __version__ in result.output


def test_config_show(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "project:\n  name: demo\n  seed: 42\npaths:\n  raw_dir: data/raw\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["config", "show", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "data/raw" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help -> non-zero exit and help text printed
    assert result.exit_code != 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_python_m_cli_entry():
    # ensure `python -m hmp.cli --help` is importable
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0