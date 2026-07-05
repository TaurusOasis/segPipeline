"""Matting quality evaluation: MQE placeholder + rule-based QA (pipeline step 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import read_binary_mask, write_uint8_image
from ..schemas import MqeRecord

log = get_logger("hmp.eval.mqe")


def _boundary_band(mask: np.ndarray, width: int = 3) -> np.ndarray:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=max(1, width))
    ero = cv2.erode(m, kernel, iterations=max(1, width))
    return (dil > 0) & (ero == 0)


def rule_based_qa(
    *,
    alpha: np.ndarray,
    mask: np.ndarray,
    prev_alpha: Optional[np.ndarray] = None,
    min_core_fill: float = 0.85,
    max_temporal_delta: float = 0.18,
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Return scores, pixel-wise reliable map, and failed rule names."""
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask).astype(bool)
    failed: list[str] = []

    core = mask
    core_alpha = alpha[core] if core.any() else np.array([0.0])
    core_fill = float(np.mean(core_alpha))
    core_score = core_fill
    if core_fill < min_core_fill:
        failed.append("core_hole")

    band = _boundary_band(mask)
    if band.any():
        edge_alignment = float(np.mean(np.abs(alpha[band] - mask[band].astype(np.float32))))
        edge_score = max(0.0, 1.0 - edge_alignment * 2.0)
    else:
        edge_score = 1.0
    if edge_score < 0.55:
        failed.append("edge_alignment")

    temporal_score = 1.0
    if prev_alpha is not None:
        prev = np.clip(np.asarray(prev_alpha, dtype=np.float32), 0.0, 1.0)
        delta = float(np.mean(np.abs(alpha - prev)))
        temporal_score = max(0.0, 1.0 - delta / max(max_temporal_delta, 1e-6))
        if delta > max_temporal_delta:
            failed.append("temporal_flicker")

    instance_score = 1.0 if core_fill > 0.2 else 0.0
    if instance_score < 0.5:
        failed.append("instance_swap")

    scores = {
        "core_score": core_score,
        "boundary_score": edge_score,
        "temporal_score": temporal_score,
        "instance_score": instance_score,
    }
    clip_quality = float(np.mean(list(scores.values())))

    reliable = np.zeros_like(alpha, dtype=np.float32)
    reliable[core] = core_score
    reliable[band] = edge_score
    if prev_alpha is not None:
        reliable = np.minimum(reliable, temporal_score)

    return scores, reliable, failed


def evaluate_instance(
    *,
    item_id: str,
    instance_id: str,
    alpha_path: Path,
    mask_path: Path,
    output_reliable_path: Path,
    output_eval_map_path: Path,
    prev_alpha_path: Optional[Path] = None,
    dry_run: bool = False,
) -> MqeRecord:
    if dry_run:
        return MqeRecord(item_id=item_id, instance_id=instance_id, review_required=True)

    from PIL import Image

    alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float32) / 255.0
    mask = read_binary_mask(mask_path)
    prev = None
    if prev_alpha_path and prev_alpha_path.exists():
        prev = np.asarray(Image.open(prev_alpha_path).convert("L"), dtype=np.float32) / 255.0

    scores, reliable, failed = rule_based_qa(alpha=alpha, mask=mask, prev_alpha=prev)
    eval_map = np.zeros_like(alpha, dtype=np.uint8)
    eval_map[reliable < 0.35] = 255
    eval_map[(reliable >= 0.35) & (reliable < 0.65)] = 128

    output_reliable_path.parent.mkdir(parents=True, exist_ok=True)
    output_eval_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_uint8_image(output_reliable_path, (reliable * 255).astype(np.uint8))
    write_uint8_image(output_eval_map_path, eval_map)

    return MqeRecord(
        item_id=item_id,
        instance_id=instance_id,
        reliable_map_path=str(output_reliable_path),
        eval_map_path=str(output_eval_map_path),
        clip_quality_score=float(np.mean(list(scores.values()))),
        scores=scores,
        review_required=bool(failed),
        failed_rules=failed,
    )


def evaluate_from_config(cfg: Config, *, project_root: Optional[Path] = None, dry_run: bool = False) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    alpha_dir = resolve_path(root, paths.get("alpha_dir", "data/alpha"))
    out_path = resolve_path(root, cfg.get("mqe", {}).get("output_path", str(alpha_dir / "mqe_report.jsonl")))

    if dry_run:
        log.info("[dry-run] would write MQE report to %s", out_path)
        return out_path

    from ..common.jsonl import write_jsonl

    # Placeholder batch: evaluate fused alphas when present.
    fused_dir = alpha_dir / "fused"
    records: list[MqeRecord] = []
    if fused_dir.exists():
        for alpha_path in sorted(fused_dir.glob("*_alpha.png")):
            stem = alpha_path.stem.replace("_alpha", "")
            mask_path = alpha_dir / ".." / "masks_refined" / f"{stem}.png"
            if not mask_path.exists():
                continue
            rec = evaluate_instance(
                item_id=stem.split("_")[0],
                instance_id="_".join(stem.split("_")[1:]),
                alpha_path=alpha_path,
                mask_path=mask_path,
                output_reliable_path=alpha_dir / "reliable" / f"{stem}_reliable.png",
                output_eval_map_path=alpha_dir / "eval_maps" / f"{stem}_eval.png",
            )
            records.append(rec)

    write_jsonl(out_path, records, overwrite=True)
    log.info("Wrote %d MQE records to %s", len(records), out_path)
    return out_path
