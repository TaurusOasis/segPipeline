"""Tests for the 12-stage pipeline registry."""

from __future__ import annotations

from hmp.pipeline.stages import PIPELINE_STAGES, build_step_plan, stage_by_index, stage_by_name


def test_pipeline_has_twelve_stages():
    assert len(PIPELINE_STAGES) == 12
    assert [s.index for s in PIPELINE_STAGES] == list(range(12))


def test_stage_lookup():
    s0 = stage_by_index(0)
    assert s0.name == "data_source_sampling"
    assert stage_by_name("final_label_output").index == 11


def test_build_step_plan_completed_through():
    steps = build_step_plan(completed_through=4, ready_at=5)
    assert len(steps) == 12
    assert steps[4].status == "done"
    assert steps[5].status == "ready"
    assert steps[6].status == "pending"
