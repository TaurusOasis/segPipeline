"""Multi-teacher alpha branch contracts (pipeline step 7).

This module does not run external matting models. It defines branch naming,
planned output paths, and dry-run command planning for Bv/Bi/Bd/Bs teachers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import AlphaBranchRecord

log = get_logger("hmp.matting.alpha_branches")

AlphaBranch = Literal["Bv", "Bi", "Bd", "Bs", "Bg"]

BRANCH_DEFAULTS: dict[str, dict[str, str]] = {
    "Bv": {
        "role": "video_matting",
        "objective": "temporal stability",
        "providers": "RVM,MatAnyone,VideoMaMa,internal_video_teacher",
    },
    "Bi": {
        "role": "image_matting",
        "objective": "hair_fingers_clothing_boundary",
        "providers": "SEMat,ViTMatte,MatteAnything,HHM_teacher",
    },
    "Bd": {
        "role": "diffusion_refine",
        "objective": "motion_blur_semitransparent_complex_edge",
        "providers": "VideoMaMa,DiffMatte,DiffusionMat,SDMatte,internal_diffusion_teacher",
    },
    "Bg": {
        "role": "generative_mask_to_matte",
        "objective": "legacy_alias_of_Bd",
        "providers": "GVM,generative_teacher",
    },
    "Bs": {
        "role": "segmentation_core",
        "objective": "semantic_body_completeness",
        "providers": "COCONut,COCO-ReM,HQ-SAM,refined_mask",
    },
}


def branch_alpha_path(alpha_dir: Path, task_id: str, branch: AlphaBranch) -> Path:
    return alpha_dir / "branches" / branch / f"{task_id}_alpha.png"


def plan_branch_outputs(
    *,
    task_id: str,
    alpha_dir: Path,
    branches: Optional[list[str]] = None,
) -> dict[str, str]:
    branches = branches or ["Bv", "Bi", "Bd", "Bs"]
    return {branch: str(branch_alpha_path(alpha_dir, task_id, branch)) for branch in branches}  # type: ignore[arg-type]


def plan_alpha_teacher_command(
    cfg: Config,
    *,
    branch: AlphaBranch,
    image_path: str,
    mask_path: str,
    trimap_path: Optional[str],
    output_path: str,
) -> str:
    """Return a dry-run command template for one branch teacher."""
    branch_cfg = cfg.get("alpha_branches", {}).get(branch, {})
    if isinstance(branch_cfg, dict):
        template = branch_cfg.get("command")
        if template:
            return str(template).format(
                branch=branch,
                image_path=image_path,
                mask_path=mask_path,
                trimap_path=trimap_path or "",
                output_path=output_path,
            )
    provider = BRANCH_DEFAULTS[branch]["providers"].split(",")[0]
    return (
        f"python external/matting_teacher/run_{branch.lower()}.py "
        f"--provider {provider} --image {image_path} --mask {mask_path} "
        f"--trimap {trimap_path or 'none'} --output {output_path}"
    )


def branch_record(branch: AlphaBranch, alpha_path: str, provider: Optional[str] = None) -> AlphaBranchRecord:
    return AlphaBranchRecord(branch=branch, alpha_path=alpha_path, provider=provider)


def resolve_branch_dir(cfg: Config, project_root: Path) -> Path:
    paths = cfg.get("paths", {})
    alpha_dir = resolve_path(project_root, paths.get("alpha_dir", "data/alpha"))
    return alpha_dir / "branches"
