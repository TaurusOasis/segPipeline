"""Alpha fusion via RL Fusion & Repair Agent (pipeline step 9)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..agents.fusion_agent import fuse_with_agent
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import write_uint8_image
from ..schemas import AlphaBranchRecord

log = get_logger("hmp.matting.alpha_fusion")


def _read_alpha(path: str | Path) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def fuse_alpha_branches(
    *,
    branches: dict[str, np.ndarray],
    reliable_map: np.ndarray,
    fg_core: np.ndarray,
    unknown_roi: np.ndarray,
    bg_core: np.ndarray,
    clip_quality: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, str], np.ndarray]:
    """Fuse branch alphas using the RL fusion agent heuristic."""
    fused, decision = fuse_with_agent(
        branches=branches,
        reliable_map=reliable_map,
        fg_core=fg_core,
        unknown_roi=unknown_roi,
        bg_core=bg_core,
        clip_quality=clip_quality,
    )
    eval_map = np.zeros_like(reliable_map, dtype=np.uint8)
    eval_map[decision.human_regions] = 255
    eval_map[(reliable_map >= 0.35) & (reliable_map < 0.65)] = 128
    dominant = {k: str(v) for k, v in decision.dominant_branches.items()}
    return fused, eval_map, dominant, decision.branch_map


def fuse_alpha_from_paths(
    cfg: Config,
    *,
    task_id: str,
    branch_records: list[AlphaBranchRecord],
    reliable_map_path: str,
    roi_paths: dict[str, str],
    output_alpha_path: Path,
    output_eval_map_path: Path,
    output_branch_source_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    root = Path(project_root) if project_root else Path.cwd()
    _ = resolve_path(root, cfg.get("paths", {}).get("alpha_dir", "data/alpha"))

    if dry_run:
        log.info("[dry-run] would fuse branches for %s -> %s", task_id, output_alpha_path)
        return output_alpha_path, output_eval_map_path

    branches = {rec.branch: _read_alpha(rec.alpha_path) for rec in branch_records}
    if Path(reliable_map_path).exists():
        reliable = _read_alpha(reliable_map_path)
    else:
        h, w = next(iter(branches.values())).shape
        reliable = np.full((h, w), 0.9, dtype=np.float32)
        reliable_dir = Path(reliable_map_path).parent
        reliable_dir.mkdir(parents=True, exist_ok=True)
        write_uint8_image(reliable_map_path, (reliable * 255).astype(np.uint8))
    fg_core = _read_alpha(roi_paths["foreground_core"]) > 0.5
    unknown = _read_alpha(roi_paths["unknown_roi"]) > 0.5
    bg_core = _read_alpha(roi_paths["background_core"]) > 0.5

    fused, eval_map, dominant, branch_map = fuse_alpha_branches(
        branches=branches,
        reliable_map=reliable,
        fg_core=fg_core,
        unknown_roi=unknown,
        bg_core=bg_core,
        clip_quality=float(np.mean(reliable)),
    )
    output_alpha_path.parent.mkdir(parents=True, exist_ok=True)
    output_eval_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_uint8_image(output_alpha_path, (fused * 255).astype(np.uint8))
    write_uint8_image(output_eval_map_path, eval_map)
    if output_branch_source_path is not None:
        output_branch_source_path.parent.mkdir(parents=True, exist_ok=True)
        write_uint8_image(output_branch_source_path, branch_map)
    log.info("Fused alpha written to %s", output_alpha_path)
    return output_alpha_path, output_eval_map_path
