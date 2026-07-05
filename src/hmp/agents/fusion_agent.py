"""Heuristic / mock RL Fusion & Repair Agent (pipeline step 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FusionBranch = Literal["Bv", "Bi", "Bd", "Bs", "reject", "human"]


@dataclass(frozen=True)
class FusionDecision:
    branch_map: np.ndarray  # uint8 enum per pixel, stored separately on disk
    reject_clip: bool
    human_regions: np.ndarray
    dominant_branches: dict[str, int]
    policy: str = "heuristic_v1"


BRANCH_CODES = {
    "background": 0,
    "Bv": 1,
    "Bi": 2,
    "Bd": 3,
    "Bs": 4,
    "human": 255,
    "reject": 254,
}


def fuse_with_agent(
    *,
    branches: dict[str, np.ndarray],
    reliable_map: np.ndarray,
    fg_core: np.ndarray,
    unknown_roi: np.ndarray,
    bg_core: np.ndarray,
    clip_quality: float,
    reject_threshold: float = 0.35,
) -> tuple[np.ndarray, FusionDecision]:
    """Decide per-region branch usage and produce fused alpha."""
    h, w = next(iter(branches.values())).shape
    fused = np.zeros((h, w), dtype=np.float32)
    branch_map = np.zeros((h, w), dtype=np.uint8)
    human = np.zeros((h, w), dtype=bool)

    if "Bv" in branches and fg_core.any():
        fused[fg_core] = branches["Bv"][fg_core]
        branch_map[fg_core] = BRANCH_CODES["Bv"]
    elif "Bs" in branches and fg_core.any():
        fused[fg_core] = branches["Bs"][fg_core]
        branch_map[fg_core] = BRANCH_CODES["Bs"]

    boundary_order = ("Bi", "Bd", "Bv", "Bs")
    for branch in boundary_order:
        if branch not in branches:
            continue
        region = unknown_roi & (branch_map == 0)
        if region.any():
            fused[region] = branches[branch][region]
            branch_map[region] = BRANCH_CODES[branch]

    if bg_core.any():
        fused[bg_core] = 0.0

    low = reliable_map < reject_threshold
    human = low | (branch_map == 0)
    branch_map[human] = BRANCH_CODES["human"]

    reject_clip = float(clip_quality) < reject_threshold
    dominant: dict[str, int] = {}
    for name, code in BRANCH_CODES.items():
        count = int(np.sum(branch_map == code))
        if count:
            dominant[name] = count

    decision = FusionDecision(
        branch_map=branch_map,
        reject_clip=reject_clip,
        human_regions=human,
        dominant_branches=dominant,
    )
    return fused, decision
