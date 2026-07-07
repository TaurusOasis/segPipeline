"""Default command-template catalog + ready-adapter factory.

Builds on :mod:`hmp.adapters.base`. For each integration in
``configs/reference_integrations.yaml`` we declare a default argv command
template (placeholders ``{input_<k>}`` / ``{output_<k>}`` / ``{param_<k>}``)
and the input/output keys it expects, so :func:`build_adapter` can assemble a
ready :class:`SubprocessAdapter` from the registry spec + template in one
call.

These templates are the *default invocation shape* for a standard checkout of
each external repo. They are meant to be tweaked (path to the repo's CLI
entrypoint, venv/conda env, weights path) via ``command_template`` overrides
or ``env`` when the real adapter is wired in a GPU env. The contract
(dry-run, output validation, provenance) is exercised without the repo
present, which is what the CPU test path checks.

Why a separate module rather than one file per ``adapter_target``: the
registry already pins the per-integration ``adapter_target`` path for the
future fine-grained subclasses (e.g. ``src/hmp/adapters/vos/sam2.py`` with an
in-process Ultralytics fast path). This catalog is the generic-subprocess
baseline those subclasses fall back to.
"""

from __future__ import annotations

from typing import Mapping, Optional

from .base import (
    AdapterRegistry,
    ExternalAdapter,
    SubprocessAdapter,
    load_registry,
)

__all__ = [
    "DEFAULT_COMMAND_TEMPLATES",
    "ADAPTER_INPUT_KEYS",
    "ADAPTER_OUTPUT_KEYS",
    "build_adapter",
    "dry_run_adapter",
    "template_for",
]


# Default argv templates per integration name. Keys mirror
# configs/reference_integrations.yaml `integrations:`. The leading
# `{param_repo_python}` placeholder lets the caller point at the right
# Python interpreter for the external repo's env (e.g. a yolo26-cu133
# venv); defaults to "python" via :func:`build_adapter`.
DEFAULT_COMMAND_TEMPLATES: dict[str, list[str]] = {
    "sam2": [
        "{param_repo_python}", "-m", "sam2.video_predict",
        "--image-dir", "{input_image_dir}",
        "--keyframe-box", "{input_keyframe_box}",
        "--output-dir", "{output_masklet_dir}",
        "--track-id-json", "{output_track_id}",
    ],
    "grounded_sam2": [
        "{param_repo_python}", "-m", "grounded_sam2.detect",
        "--image", "{input_image}",
        "--text-prompt", "{param_text_prompt}",
        "--output-candidates", "{output_person_candidates}",
        "--output-bbox", "{output_bbox}",
        "--output-rle", "{output_rle_mask}",
        "--output-score", "{output_score}",
    ],
    "groundingdino": [
        "{param_repo_python}", "-m", "groundingdino.predict",
        "--image", "{input_image}",
        "--text-prompt", "{param_text_prompt}",
        "--output-bbox", "{output_bbox}",
        "--output-score", "{output_score}",
        "--output-phrase", "{output_phrase}",
    ],
    "ultralytics_yolo": [
        "{param_repo_python}", "-m", "hmp.yolo.detect_cli",
        "--image", "{input_image}",
        "--weights", "{param_weights}",
        "--output-bbox", "{output_bbox}",
        "--output-mask", "{output_mask}",
        "--output-score", "{output_score}",
    ],
    "cutie": [
        "{param_repo_python}", "-m", "cutie.inference",
        "--image-dir", "{input_image_dir}",
        "--mask-prompt", "{input_mask_prompt}",
        "--output", "{output_masklet}",
        "--track-id-json", "{output_track_id}",
    ],
    "xmem": [
        "{param_repo_python}", "-m", "xmem.infer",
        "--image-dir", "{input_image_dir}",
        "--mask-prompt", "{input_mask_prompt}",
        "--output", "{output_masklet}",
        "--track-id-json", "{output_track_id}",
    ],
    "samrefiner": [
        "{param_repo_python}", "-m", "samrefiner.refine",
        "--image", "{input_image}",
        "--coarse-mask", "{input_coarse_mask}",
        "--output", "{output_refined_mask}",
        "--quality-json", "{output_mask_quality}",
    ],
    "hq_sam": [
        "{param_repo_python}", "-m", "sam_hq.predict",
        "--image", "{input_image}",
        "--box", "{input_box}",
        "--output", "{output_refined_mask}",
    ],
    "cascadepsp": [
        "{param_repo_python}", "-m", "cascadepsp.refine",
        "--image", "{input_image}",
        "--coarse-mask", "{input_coarse_mask}",
        "--output", "{output_refined_mask}",
    ],
    "matanyone": [
        "{param_repo_python}", "-m", "matanyone.infer",
        "--image-dir", "{input_image_dir}",
        "--target-mask", "{input_target_mask}",
        "--output", "{output_alpha_video}",
        "--branch-source", "{output_branch_source}",
    ],
    "matanyone2": [
        "{param_repo_python}", "-m", "matanyone2.infer",
        "--image-dir", "{input_image_dir}",
        "--target-mask", "{input_target_mask}",
        "--output", "{output_alpha_video}",
        "--eval-map", "{output_eval_map}",
        "--quality-json", "{output_quality_score}",
    ],
    "maggie": [
        "{param_repo_python}", "-m", "maggie.infer",
        "--image", "{input_image}",
        "--instance-mask", "{input_instance_mask}",
        "--output", "{output_alpha}",
        "--instance-output", "{output_instance_alpha}",
    ],
    "semat": [
        "{param_repo_python}", "-m", "semat.infer",
        "--image", "{input_image}",
        "--person-mask", "{input_person_mask}",
        "--output", "{output_alpha_image}",
    ],
    "matting_anything": [
        "{param_repo_python}", "-m", "mam.infer",
        "--image", "{input_image}",
        "--mask", "{input_mask}",
        "--output", "{output_alpha_image}",
    ],
    "rvm": [
        "{param_repo_python}", "-m", "rvm.infer",
        "--image-dir", "{input_image_dir}",
        "--output", "{output_alpha_video}",
    ],
    "videomama": [
        "{param_repo_python}", "-m", "videomama.refine",
        "--image-dir", "{input_image_dir}",
        "--coarse-alpha-dir", "{input_coarse_alpha_dir}",
        "--roi", "{input_roi}",
        "--output", "{output_alpha_diffusion}",
        "--refine-roi", "{output_refine_roi}",
    ],
    "diffmatte": [
        "{param_repo_python}", "-m", "diffmatte.infer",
        "--image", "{input_image}",
        "--mask", "{input_mask}",
        "--output", "{output_alpha_diffusion}",
    ],
    "sdmatte": [
        "{param_repo_python}", "-m", "sdmatte.infer",
        "--image", "{input_image}",
        "--output", "{output_alpha_diffusion}",
    ],
    "raft": [
        "{param_repo_python}", "-m", "raft.compute_flow",
        "--prev", "{input_prev_alpha}",
        "--cur", "{input_cur_alpha}",
        "--error-json", "{output_temporal_error}",
        "--consistency-json", "{output_flow_consistency_score}",
    ],
    "gmflow": [
        "{param_repo_python}", "-m", "gmflow.compute_flow",
        "--prev", "{input_prev_alpha}",
        "--cur", "{input_cur_alpha}",
        "--error-json", "{output_temporal_error}",
        "--consistency-json", "{output_flow_consistency_score}",
    ],
    "mmagic": [
        "{param_repo_python}", "-m", "mmagic.matting_metrics",
        "--pred-dir", "{input_pred_dir}",
        "--gt-dir", "{input_gt_dir}",
        "--trimap-dir", "{input_trimap_dir}",
        "--output-sad", "{output_sad}",
        "--output-mse", "{output_mse}",
        "--output-gradient", "{output_gradient}",
        "--output-connectivity", "{output_connectivity}",
    ],
    "fiftyone": [
        "{param_repo_python}", "-m", "hmp.adapters.hitl.fiftyone_view",
        "--dataset-dir", "{input_dataset_dir}",
        "--view-spec", "{param_view_spec}",
        "--dataset-view", "{output_dataset_view}",
        "--export", "{output_review_selection}",
    ],
    "cvat": [
        "{param_repo_python}", "-m", "hmp.adapters.hitl.cvat_bridge",
        "--task-id", "{param_task_id}",
        "--export", "{output_human_edits}",
        "--corrected-prompts", "{output_corrected_prompts}",
        "--audit-log", "{output_audit_log}",
    ],
    "label_studio": [
        "{param_repo_python}", "-m", "hmp.adapters.hitl.label_studio_bridge",
        "--project-id", "{param_project_id}",
        "--export", "{output_human_edits}",
        "--audit-log", "{output_audit_log}",
    ],
    "gymnasium": [
        "{param_repo_python}", "-m", "hmp.adapters.active_labeling.gym_env",
        "--config", "{input_env_config}",
        "--episode-out", "{output_agent_episode}",
        "--reward-trace", "{output_reward_trace}",
    ],
    "stable_baselines3": [
        "{param_repo_python}", "-m", "hmp.adapters.active_labeling.sb3_agent",
        "--env-config", "{input_env_config}",
        "--checkpoint", "{output_policy_checkpoint}",
        "--trace", "{output_decision_trace}",
    ],
}

# Documented input keys per integration (for callers / dry-run previews).
ADAPTER_INPUT_KEYS: dict[str, list[str]] = {
    "sam2": ["image_dir", "keyframe_box"],
    "samrefiner": ["image", "coarse_mask"],
    "hq_sam": ["image", "box"],
    "cutie": ["image_dir", "mask_prompt"],
    "xmem": ["image_dir", "mask_prompt"],
    "matanyone": ["image_dir", "target_mask"],
    "maggie": ["image", "instance_mask"],
    "semat": ["image", "person_mask"],
    "raft": ["prev_alpha", "cur_alpha"],
    "gmflow": ["prev_alpha", "cur_alpha"],
    "videomama": ["image_dir", "coarse_alpha_dir", "roi"],
    "grounded_sam2": ["image"],
    "groundingdino": ["image"],
    "ultralytics_yolo": ["image"],
    "mmagic": ["pred_dir", "gt_dir", "trimap_dir"],
    "fiftyone": ["dataset_dir"],
    "gymnasium": ["env_config"],
    "stable_baselines3": ["env_config"],
}

# Documented output keys per integration (subset; aligns with registry
# expected_outputs where the output is a single file/dir we can validate).
ADAPTER_OUTPUT_KEYS: dict[str, list[str]] = {
    "sam2": ["masklet_dir", "track_id"],
    "samrefiner": ["refined_mask", "mask_quality"],
    "hq_sam": ["refined_mask"],
    "cutie": ["masklet", "track_id"],
    "xmem": ["masklet", "track_id"],
    "matanyone": ["alpha_video", "branch_source"],
    "matanyone2": ["alpha_video", "eval_map", "quality_score"],
    "maggie": ["alpha", "instance_alpha"],
    "semat": ["alpha_image"],
    "rvm": ["alpha_video"],
    "matting_anything": ["alpha_image"],
    "raft": ["temporal_error", "flow_consistency_score"],
    "gmflow": ["temporal_error", "flow_consistency_score"],
    "videomama": ["alpha_diffusion", "refine_roi"],
    "diffmatte": ["alpha_diffusion"],
    "sdmatte": ["alpha_diffusion"],
    "cascadepsp": ["refined_mask"],
    "grounded_sam2": ["person_candidates", "bbox", "rle_mask", "score"],
    "groundingdino": ["bbox", "score", "phrase"],
    "ultralytics_yolo": ["bbox", "mask", "score"],
    "mmagic": ["sad", "mse", "gradient", "connectivity"],
    "fiftyone": ["dataset_view", "review_selection"],
    "cvat": ["human_edits", "corrected_prompts", "audit_log"],
    "label_studio": ["human_edits", "audit_log"],
    "gymnasium": ["agent_episode", "reward_trace"],
    "stable_baselines3": ["policy_checkpoint", "decision_trace"],
}


def template_for(name: str) -> list[str]:
    """Return the default command template for ``name`` (raises if absent)."""
    try:
        return list(DEFAULT_COMMAND_TEMPLATES[name])
    except KeyError:
        raise KeyError(f"no default command template registered for {name!r}") from None


def build_adapter(
    name: str,
    workdir: str | None = None,
    *,
    registry: Optional[AdapterRegistry] = None,
    command_template: Optional[list[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: float = 600.0,
) -> SubprocessAdapter:
    """Assemble a ready :class:`SubprocessAdapter` for integration ``name``.

    Pulls the spec from the registry and the default command template from
    :data:`DEFAULT_COMMAND_TEMPLATES` (unless ``command_template`` overrides
    it). ``workdir`` defaults to a per-name subdir under the cwd (created on
    first run); pass an explicit path for deterministic test placement.
    """
    reg = registry or load_registry()
    spec = reg.get(name)
    tmpl = list(command_template) if command_template is not None else template_for(name)
    env_overlay = dict(env or {})
    env_overlay.setdefault("REPO_PYTHON", env_overlay.get("REPO_PYTHON", "python"))
    workdir = workdir or f"runs/adapters/{name}"
    return SubprocessAdapter(
        spec,
        workdir=workdir,
        command_template=tmpl,
        env=env_overlay,
        timeout_s=timeout_s,
    )


def dry_run_adapter(
    name: str,
    workdir: str | None = None,
    *,
    inputs: Optional[Mapping[str, str]] = None,
    outputs: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, object]] = None,
    registry: Optional[AdapterRegistry] = None,
    command_template: Optional[list[str]] = None,
    env: Optional[Mapping[str, str]] = None,
):
    """Convenience: build the adapter and return its :class:`AdapterResult` dry run."""
    adapter = build_adapter(
        name,
        workdir=workdir,
        registry=registry,
        command_template=command_template,
        env=env,
    )
    # Supply the default repo python param if the template references it and
    # the caller did not.
    full_params = dict(params or {})
    if "{param_repo_python}" in " ".join(adapter.command_template):
        full_params.setdefault("repo_python", adapter.env_overlay.get("REPO_PYTHON", "python"))
    return adapter.dry_run(inputs or {}, outputs or {}, params=full_params)