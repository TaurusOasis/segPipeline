"""Alpha teacher adapters for Bv/Bi/Bd/Bs branches (pipeline step 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..common.logging import get_logger
from ..common.subprocess_utils import render_command, run_command, validate_outputs
from ..config import Config, resolve_path
from ..data.mask_io import read_binary_mask, write_uint8_image
from ..matting.alpha_branches import AlphaBranch, BRANCH_DEFAULTS, branch_alpha_path, plan_alpha_teacher_command
from ..schemas import AlphaBranchRecord

log = get_logger("hmp.matting.alpha_teacher")


def _feather_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.float32)
    if sigma <= 0:
        return m
    k = int(max(3, round(sigma * 4)) | 1)
    blurred = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def mock_branch_alpha(mask: np.ndarray, branch: AlphaBranch) -> np.ndarray:
    """CPU-only synthetic alpha for tests and demo pipeline."""
    if branch == "Bs":
        return (np.asarray(mask) > 0).astype(np.float32)
    if branch == "Bv":
        return _feather_alpha(mask, sigma=1.0)
    if branch == "Bi":
        return _feather_alpha(mask, sigma=2.5)
    if branch in {"Bd", "Bg"}:
        return _feather_alpha(mask, sigma=4.0)
    raise ValueError(f"unknown branch: {branch}")


def generate_branch_alpha(
    cfg: Config,
    *,
    branch: AlphaBranch,
    image_path: str,
    mask_path: str,
    trimap_path: Optional[str],
    output_path: Path,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    provider: str = "mock",
) -> AlphaBranchRecord:
    root = Path(project_root) if project_root else Path.cwd()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "mock":
        if dry_run:
            log.info("[dry-run] mock %s alpha -> %s", branch, output_path)
            return AlphaBranchRecord(branch=branch, alpha_path=str(output_path), provider="mock")
        mask = read_binary_mask(mask_path)
        alpha = mock_branch_alpha(mask, branch)
        write_uint8_image(output_path, (alpha * 255).astype(np.uint8))
        return AlphaBranchRecord(branch=branch, alpha_path=str(output_path), provider="mock")

    command = plan_alpha_teacher_command(
        cfg,
        branch=branch,
        image_path=image_path,
        mask_path=mask_path,
        trimap_path=trimap_path,
        output_path=str(output_path),
    )
    branch_cfg = cfg.get("alpha_branches", {}).get(branch, {})
    if isinstance(branch_cfg, dict) and branch_cfg.get("command_template"):
        command = render_command(str(branch_cfg["command_template"]), {
            "branch": branch,
            "image_path": image_path,
            "mask_path": mask_path,
            "trimap_path": trimap_path or "",
            "output_path": str(output_path),
        })
    run_command(command, cwd=root, dry_run=dry_run)
    if not dry_run:
        validate_outputs([output_path])
    return AlphaBranchRecord(branch=branch, alpha_path=str(output_path), provider=provider)


def generate_all_branch_alphas(
    cfg: Config,
    *,
    task_id: str,
    image_path: str,
    mask_path: str,
    trimap_path: Optional[str],
    alpha_dir: Path,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    provider: str = "mock",
    branches: Optional[list[AlphaBranch]] = None,
) -> list[AlphaBranchRecord]:
    branches = branches or ["Bv", "Bi", "Bd", "Bs"]
    records: list[AlphaBranchRecord] = []
    for branch in branches:
        out = branch_alpha_path(alpha_dir, task_id, branch)
        records.append(
            generate_branch_alpha(
                cfg,
                branch=branch,
                image_path=image_path,
                mask_path=mask_path,
                trimap_path=trimap_path,
                output_path=out,
                project_root=project_root,
                dry_run=dry_run,
                provider=provider,
            )
        )
    return records
