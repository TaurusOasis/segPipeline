"""Tests for COCONut benchmark and mock auto-labeling."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from hmp.cli import app
from hmp.common.jsonl import read_jsonl_list
from hmp.eval.coconut_benchmark import run_coconut_benchmark
from hmp.labeling.mock_sam2 import segment_with_prompts
from hmp.agents.prompt_agent import plan_prompts
from hmp.schemas import BenchmarkRecord

runner = CliRunner()


def _cfg(tmp_path: Path, limit: int = 4) -> Path:
    cfg = {
        "project": {"seed": 42},
        "local_postprocess": {"remove_small_components": True, "min_component_area": 8, "fill_holes": True},
        "coconut_benchmark": {
            "coconut_root": "/home/genesis/Train/Dataset/coconut",
            "image_root": "/home/genesis/Train/Dataset/coco2017",
            "json_path": "/home/genesis/Train/Dataset/coconut/relabeled_coco_val.json",
            "mask_dir": "/home/genesis/Train/Dataset/coconut/relabeled_coco_val",
            "image_subdir": "val2017",
            "limit": limit,
            "seed": 42,
            "detector_mode": "gt_bbox",
            "sam_mode": "oracle",
            "output_dir": str(tmp_path / "benchmark"),
        },
    }
    p = tmp_path / "coconut_benchmark.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_prompt_agent_and_oracle_sam_match_gt():
    import numpy as np

    gt = np.zeros((64, 64), dtype=bool)
    gt[10:50, 10:40] = True
    decision = plan_prompts(bbox_xyxy=[10, 10, 40, 50], width=64, height=64, gt_mask=gt)
    pred = segment_with_prompts(np.zeros((64, 64, 3), dtype=np.uint8), decision, gt_mask=gt)
    assert pred.shape == gt.shape
    assert pred.sum() == gt.sum()


def test_coconut_benchmark_runs_on_real_subset(tmp_path):
    cfg_path = _cfg(tmp_path, limit=4)
    import json

    from hmp.config import load_config

    cfg = load_config(cfg_path)
    out = run_coconut_benchmark(cfg, project_root=tmp_path)
    assert out.exists()
    summary = json.loads((tmp_path / "benchmark" / "benchmark_summary.json").read_text())
    assert summary["instances"] >= 1
    assert summary["mean_mask_iou"] >= 0.99
    assert summary["decision_counts"]["accept"] >= 1
    assert "error_buckets" in summary
    assert "recommendations" in summary
    records = read_jsonl_list(out, model=BenchmarkRecord)
    assert records[0].pred_mask_path
    assert records[0].gt_mask_path
    assert records[0].diff_mask_path
    assert Path(records[0].pred_mask_path).exists()
    assert Path(records[0].gt_mask_path).exists()
    assert Path(records[0].diff_mask_path).exists()
    assert records[0].decision == "accept"


def test_coconut_benchmark_cli(tmp_path):
    cfg_path = _cfg(tmp_path, limit=3)
    result = runner.invoke(app, ["eval", "coconut-benchmark", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
