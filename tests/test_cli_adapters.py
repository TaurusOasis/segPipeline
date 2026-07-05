"""CLI tests for the `hmp adapters` group (list + dry-run)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hmp.cli import app

runner = CliRunner()


def test_adapters_help():
    result = runner.invoke(app, ["adapters", "--help"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "list" in result.output


def test_adapters_list_all():
    result = runner.invoke(app, ["adapters", "list"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.strip().splitlines() if l]
    # The registry has 26 integrations.
    assert len(lines) >= 20
    names = {l.split("\t")[0] for l in lines}
    assert "sam2" in names
    assert "samrefiner" in names
    assert "raft" in names


def test_adapters_list_group_filter():
    result = runner.invoke(app, ["adapters", "list", "--group", "mask_refine"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.strip().splitlines() if l]
    groups = {l.split("\t")[1] for l in lines}
    assert groups == {"mask_refine"}
    names = {l.split("\t")[0] for l in lines}
    assert "samrefiner" in names
    assert "hq_sam" in names


def test_adapters_dry_run_resolves_command(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "adapters", "dry-run",
            "--name", "samrefiner",
            "--input", "image=/data/a.png",
            "--input", "coarse_mask=/data/coarse.png",
            "--output", f"refined_mask={tmp_path / 'ref.png'}",
            "--output", f"mask_quality={tmp_path / 'q.json'}",
            "--param", "repo_python=/opt/venv/bin/python",
            "--workdir", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["name"] == "samrefiner"
    assert payload["dry_run"] is True
    assert "/opt/venv/bin/python" in payload["command"]
    assert "/data/a.png" in payload["command"]
    assert payload["spec"]["group"] == "mask_refine"
    assert payload["spec"]["expected_outputs"] == ["refined_mask", "mask_quality"]
    assert payload["env"]["REPO_PYTHON"] == "python"


def test_adapters_dry_run_default_repo_python(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "adapters", "dry-run",
            "--name", "samrefiner",
            "--input", "image=a.png",
            "--input", "coarse_mask=c.png",
            "--output", f"refined_mask={tmp_path / 'r.png'}",
            "--output", f"mask_quality={tmp_path / 'q.json'}",
            "--workdir", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # repo_python defaulted to "python" since not supplied.
    assert payload["command"][0] == "python"


def test_adapters_dry_run_unknown_integration(tmp_path: Path):
    result = runner.invoke(
        app,
        ["adapters", "dry-run", "--name", "nope", "--workdir", str(tmp_path)],
    )
    assert result.exit_code != 0


def test_adapters_dry_run_missing_input_placeholder(tmp_path: Path):
    # coarse_mask input omitted -> MissingPlaceholder -> non-zero exit.
    result = runner.invoke(
        app,
        [
            "adapters", "dry-run",
            "--name", "samrefiner",
            "--input", "image=a.png",
            "--output", f"refined_mask={tmp_path / 'r.png'}",
            "--output", f"mask_quality={tmp_path / 'q.json'}",
            "--workdir", str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_adapters_dry_run_command_template_override(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "adapters", "dry-run",
            "--name", "hq_sam",
            "--input", "image=a.png",
            "--input", "box=10,10,40,50",
            "--output", f"refined_mask={tmp_path / 'r.png'}",
            "--command-template", "echo,{input_image},{input_box},{output_refined_mask}",
            "--workdir", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == ["echo", "a.png", "10,10,40,50", str(tmp_path / "r.png")]


def test_adapters_dry_run_bad_kv_format(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "adapters", "dry-run",
            "--name", "samrefiner",
            "--input", "no_equals_here",
            "--workdir", str(tmp_path),
        ],
    )
    assert result.exit_code != 0