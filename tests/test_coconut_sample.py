"""Tests for COCONut manifest sampling (stage 0)."""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from hmp.cli import app
from hmp.config import load_config
from hmp.data.coconut_sample import build_coconut_manifest


runner = CliRunner()


def test_build_coconut_manifest_dry_run(tmp_path):
    cfg_path = tmp_path / "sample.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "coconut_sample": {
                    "manifest_path": str(tmp_path / "manifest.jsonl"),
                    "limit": 8,
                },
                "coconut_benchmark": {
                    "coconut_root": "/home/genesis/Train/Dataset/coconut",
                    "image_root": "/home/genesis/Train/Dataset/coco2017",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    out = build_coconut_manifest(cfg, project_root=tmp_path, dry_run=True)
    assert out == tmp_path / "manifest.jsonl"
    assert not out.exists()


def test_coconut_sample_cli_dry_run(tmp_path):
    cfg_path = tmp_path / "sample.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "coconut_sample": {"manifest_path": str(tmp_path / "manifest.jsonl"), "limit": 4},
                "coconut_benchmark": {
                    "coconut_root": "/home/genesis/Train/Dataset/coconut",
                    "image_root": "/home/genesis/Train/Dataset/coco2017",
                },
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "coconut-sample", "--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0, result.output
