"""COCONut auto-label benchmark: compare pipeline masks against GT labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.coconut_io import iter_coconut_person_samples, sample_to_media_item
from ..data.mask_io import mask_to_bbox_xyxy, write_binary_mask, write_uint8_image
from ..eval.label_quality import (
    decision_and_tags,
    mask_error_stats,
    quality_gates_from_config,
)
from ..labeling.auto_label_core import label_instance_from_bbox, labeling_runtime_from_config
from ..labeling.yolo_person_detector import PersonDetection, bbox_iou, detect_persons, match_detection_for_gt
from ..schemas import AnnotationRecord, BenchmarkRecord, InstanceAnnotation, MediaItem

log = get_logger("hmp.eval.coconut_benchmark")

DetectorMode = Literal["gt_bbox", "jitter_bbox", "center_prior", "yolo_person"]
SamMode = Literal["grabcut", "oracle", "noisy_oracle", "sam2"]

_quality_gates = quality_gates_from_config
_decision_and_tags = decision_and_tags
_mask_error_stats = mask_error_stats


def _jitter_bbox(bbox: list[int], width: int, height: int, scale: float = 0.08, rng: np.random.Generator | None = None) -> list[int]:
    rng = rng or np.random.default_rng(0)
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx = int(round(bw * scale * rng.uniform(-1, 1)))
    dy = int(round(bh * scale * rng.uniform(-1, 1)))
    nx1 = max(0, x1 + dx)
    ny1 = max(0, y1 + dy)
    nx2 = min(width, x2 + dx)
    ny2 = min(height, y2 + dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return bbox
    return [nx1, ny1, nx2, ny2]


def _center_prior_bbox(width: int, height: int) -> list[int]:
    bw = max(1, int(width * 0.45))
    bh = max(1, int(height * 0.65))
    x1 = (width - bw) // 2
    y1 = (height - bh) // 2
    return [x1, y1, x1 + bw, y1 + bh]


def _detect_person_bbox(
    person,
    sample,
    *,
    mode: DetectorMode,
    rng: np.random.Generator,
    yolo_detections: list[PersonDetection] | None = None,
    yolo_used: set[int] | None = None,
    yolo_match_iou: float = 0.3,
) -> tuple[list[int], dict[str, float]]:
    """Return bbox for prompt planning plus optional detector metadata."""
    meta: dict[str, float] = {}
    if mode == "gt_bbox":
        return list(person.bbox_xyxy), meta
    if mode == "jitter_bbox":
        return _jitter_bbox(person.bbox_xyxy, sample.width, sample.height, rng=rng), meta
    if mode == "center_prior":
        return _center_prior_bbox(sample.width, sample.height), meta

    used = yolo_used if yolo_used is not None else set()
    detections = yolo_detections or []
    matched, match_iou = match_detection_for_gt(
        detections,
        person.bbox_xyxy,
        used_indices=used,
        iou_threshold=yolo_match_iou,
    )
    meta["det_match_iou"] = float(match_iou)
    if matched is not None:
        meta["det_matched"] = 1.0
        meta["det_score"] = float(matched.score)
        return list(matched.bbox_xyxy), meta
    meta["det_matched"] = 0.0
    return list(person.bbox_xyxy), meta


def _diff_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = np.asarray(pred) > 0
    gt_b = np.asarray(gt) > 0
    diff = np.zeros(gt_b.shape, dtype=np.uint8)
    diff[gt_b & pred_b] = 255
    diff[gt_b & ~pred_b] = 85
    diff[~gt_b & pred_b] = 170
    return diff


def _read_mask_or_none(path: str) -> np.ndarray | None:
    if not path:
        return None
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return arr > 0


def _crop_box_from_masks(masks: list[np.ndarray | None], width: int, height: int, pad_ratio: float = 0.18) -> tuple[int, int, int, int]:
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if mask is not None and mask.shape == union.shape:
            union |= mask
    if not union.any():
        return (0, 0, width, height)
    ys, xs = np.where(union)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def _resize_tile(img: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text[:64], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay_mask(img: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int]) -> np.ndarray:
    if mask is None:
        return img.copy()
    out = img.copy()
    m = mask > 0
    overlay = out.copy()
    overlay[m] = np.asarray(color, dtype=np.uint8)
    out[m] = (0.55 * out[m] + 0.45 * overlay[m]).astype(np.uint8)
    return out


def _diff_to_bgr(diff: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if diff is None:
        return out
    d = np.asarray(diff)
    out[d == 255] = (40, 190, 40)   # true positive: green
    out[d == 85] = (40, 40, 230)    # false negative: red
    out[d == 170] = (230, 140, 30)  # false positive: blue/orange
    return out


def write_benchmark_contact_sheet(
    out_dir: Path,
    *,
    records: list[BenchmarkRecord] | None = None,
    out_path: Path | None = None,
    max_items: int = 12,
    tile_width: int = 220,
) -> Path:
    """Write a worst-case visual sheet: image, GT, prediction, diff."""
    import cv2

    out_dir = Path(out_dir)
    if records is None:
        records = read_jsonl_list(out_dir / "benchmark_records.jsonl", model=BenchmarkRecord)
    selected = sorted(records, key=lambda r: (r.mask_iou, r.boundary_f_score))[:max_items]
    out_path = out_path or (out_dir / "contact_sheet_worst.png")
    tile_height = int(round(tile_width * 0.75))
    rows: list[np.ndarray] = []
    for record in selected:
        image = cv2.imread(record.image_path)
        if image is None:
            continue
        gt = _read_mask_or_none(record.gt_mask_path)
        pred = _read_mask_or_none(record.pred_mask_path)
        diff_raw = cv2.imread(record.diff_mask_path, cv2.IMREAD_GRAYSCALE) if record.diff_mask_path else None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = _crop_box_from_masks([gt, pred], w, h)
        img_crop = image[y1:y2, x1:x2]
        gt_crop = gt[y1:y2, x1:x2] if gt is not None and gt.shape[:2] == (h, w) else None
        pred_crop = pred[y1:y2, x1:x2] if pred is not None and pred.shape[:2] == (h, w) else None
        diff_crop = diff_raw[y1:y2, x1:x2] if diff_raw is not None and diff_raw.shape[:2] == (h, w) else None
        if img_crop.size == 0:
            continue
        base = _resize_tile(img_crop, tile_width, tile_height)
        gt_tile = _resize_tile(_overlay_mask(img_crop, gt_crop, (40, 190, 40)), tile_width, tile_height)
        pred_tile = _resize_tile(_overlay_mask(img_crop, pred_crop, (230, 140, 30)), tile_width, tile_height)
        diff_tile = _resize_tile(_diff_to_bgr(diff_crop, img_crop.shape[:2]), tile_width, tile_height)
        title = f"{record.item_id}/{record.instance_id} IoU={record.mask_iou:.3f} BF1={record.boundary_f_score:.3f}"
        rows.append(
            np.concatenate(
                [
                    _label_tile(base, title),
                    _label_tile(gt_tile, "GT"),
                    _label_tile(pred_tile, "PRED"),
                    _label_tile(diff_tile, "DIFF green=TP red=FN blue=FP"),
                ],
                axis=1,
            )
        )
    if not rows:
        sheet = np.zeros((tile_height, tile_width * 4, 3), dtype=np.uint8)
        sheet = _label_tile(sheet, "No visual records")
    else:
        sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _count_by(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _recommendations(error_buckets: dict[str, int], decision_counts: dict[str, int]) -> list[str]:
    recs: list[str] = []
    if error_buckets.get("detector_miss", 0):
        recs.append("Prioritize detector recall: lower YOLO confidence or add GroundingDINO/person-classifier fallback.")
    if error_buckets.get("background_leak", 0):
        recs.append("Strengthen negative prompts and identity separation around multi-person/background-leak cases.")
    if error_buckets.get("missed_foreground", 0):
        recs.append("Use prompt-agent correction: add positive points/scribbles on missed limbs, hair, or thin structures.")
    if error_buckets.get("bad_boundary", 0):
        recs.append("Send boundary ROI to HQ-SAM or Bd diffusion refine; do not train alpha on hard-mask boundary directly.")
    if error_buckets.get("small_person", 0):
        recs.append("Track small-person bucket separately; use it for detector/prompt stress tests before matting supervision.")
    if decision_counts.get("review", 0) or decision_counts.get("reject", 0):
        recs.append("Export review/reject records as the next active-labeling queue.")
    return recs or ["Current sample passes configured gates; expand COCONut limit or add harder buckets."]


def _build_summary(
    *,
    records: list[BenchmarkRecord],
    dataset: str,
    detector_mode: str,
    sam_mode: str,
    total_s: float,
    quality_gates: dict[str, float],
    worst_k: int,
) -> dict[str, object]:
    decision_counts = _count_by([r.decision for r in records])
    error_buckets = _count_by([tag for r in records for tag in r.error_tags])
    mean_iou = _mean([r.mask_iou for r in records])
    mean_bf1 = _mean([r.boundary_f_score for r in records])
    mean_ms = _mean([r.elapsed_ms for r in records])
    ips = len(records) / max(total_s, 1e-6)
    worst = sorted(records, key=lambda r: (r.mask_iou, r.boundary_f_score))[:worst_k]
    return {
        "dataset": dataset,
        "instances": len(records),
        "detector_mode": detector_mode,
        "sam_mode": sam_mode,
        "mean_mask_iou": mean_iou,
        "mean_boundary_f1": mean_bf1,
        "mean_bbox_iou": _mean([r.bbox_iou for r in records if r.bbox_iou is not None]),
        "mean_false_positive_ratio": _mean([r.false_positive_ratio or 0.0 for r in records]),
        "mean_false_negative_ratio": _mean([r.false_negative_ratio or 0.0 for r in records]),
        "mask_iou_quantiles": _quantiles([r.mask_iou for r in records]),
        "boundary_f1_quantiles": _quantiles([r.boundary_f_score for r in records]),
        "decision_counts": decision_counts,
        "accept_rate": float(decision_counts.get("accept", 0)) / float(max(len(records), 1)),
        "review_rate": float(decision_counts.get("review", 0)) / float(max(len(records), 1)),
        "reject_rate": float(decision_counts.get("reject", 0)) / float(max(len(records), 1)),
        "error_buckets": error_buckets,
        "quality_gates": quality_gates,
        "recommendations": _recommendations(error_buckets, decision_counts),
        "worst_records": [
            {
                "item_id": r.item_id,
                "instance_id": r.instance_id,
                "mask_iou": r.mask_iou,
                "boundary_f_score": r.boundary_f_score,
                "decision": r.decision,
                "error_tags": r.error_tags,
                "improvement_hint": r.improvement_hint,
                "image_path": r.image_path,
                "gt_mask_path": r.gt_mask_path,
                "pred_mask_path": r.pred_mask_path,
                "diff_mask_path": r.diff_mask_path,
            }
            for r in worst
        ],
        "mean_elapsed_ms": mean_ms,
        "instances_per_second": ips,
        "total_seconds": total_s,
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    decisions = summary.get("decision_counts", {})
    buckets = summary.get("error_buckets", {})
    worst = summary.get("worst_records", [])
    recs = summary.get("recommendations", [])
    lines = [
        "# COCONut Auto-Label Benchmark",
        "",
        f"- instances: **{summary['instances']}**",
        f"- detector: `{summary['detector_mode']}`",
        f"- sam: `{summary['sam_mode']}`",
        f"- mean mask IoU: **{float(summary['mean_mask_iou']):.4f}**",
        f"- mean boundary F1: **{float(summary['mean_boundary_f1']):.4f}**",
        f"- accept / review / reject: **{decisions.get('accept', 0)} / {decisions.get('review', 0)} / {decisions.get('reject', 0)}**",
        f"- mean FP / FN ratio: **{float(summary['mean_false_positive_ratio']):.4f} / {float(summary['mean_false_negative_ratio']):.4f}**",
        f"- mean latency: **{float(summary['mean_elapsed_ms']):.1f} ms/instance**",
        f"- throughput: **{float(summary['instances_per_second']):.2f} inst/s**",
        "",
        "## Error Buckets",
        "",
    ]
    if buckets:
        lines += ["| bucket | count |", "|---|---:|"]
        lines += [f"| `{k}` | {v} |" for k, v in buckets.items()]
    else:
        lines.append("_No error buckets._")
    lines += ["", "## Recommendations", ""]
    lines += [f"- {r}" for r in recs] if recs else ["- Expand sample size."]
    lines += ["", "## Worst Records", ""]
    if worst:
        lines += ["| item | inst | IoU | BF1 | decision | tags |", "|---|---|---:|---:|---|---|"]
        for row in worst:  # type: ignore[assignment]
            lines.append(
                f"| `{row['item_id']}` | `{row['instance_id']}` | "
                f"{float(row['mask_iou']):.4f} | {float(row['boundary_f_score']):.4f} | "
                f"`{row['decision']}` | `{','.join(row['error_tags'])}` |"
            )
    else:
        lines.append("_No records._")
    lines += ["", "Records: `benchmark_records.jsonl`", "Pred masks: `pred_masks/`", "GT binary masks: `gt_masks/`", "Diff masks: `diff_masks/`"]
    if summary.get("contact_sheet_path"):
        lines += [f"Contact sheet: `{summary['contact_sheet_path']}`"]
    return "\n".join(lines)


def run_coconut_benchmark(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    bcfg = cfg.get("coconut_benchmark", {})
    coconut_root = Path(bcfg.get("coconut_root", "/home/genesis/Train/Dataset/coconut"))
    image_root = Path(bcfg.get("image_root", "/home/genesis/Train/Dataset/coco2017"))
    json_path = Path(bcfg.get("json_path", coconut_root / "relabeled_coco_val.json"))
    mask_dir = Path(bcfg.get("mask_dir", coconut_root / "relabeled_coco_val"))
    limit = int(bcfg.get("limit", 64))
    seed = int(bcfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    detector_mode: DetectorMode = bcfg.get("detector_mode", "gt_bbox")
    sam_mode: SamMode = bcfg.get("sam_mode", "grabcut")
    noise_level = float(bcfg.get("noise_level", 0.15))
    yolo_weights = str(bcfg.get("yolo_weights", "/home/genesis/Train/Code/ultralytics/yolo26s-seg.pt"))
    yolo_conf = float(bcfg.get("yolo_conf", 0.25))
    yolo_iou = float(bcfg.get("yolo_iou", 0.7))
    yolo_match_iou = float(bcfg.get("yolo_match_iou", 0.3))
    sam2_weights = str(bcfg.get("sam2_weights", "sam2_b.pt"))
    device = bcfg.get("device", 0)
    write_masks = bool(bcfg.get("write_masks", True))
    worst_k = int(bcfg.get("worst_k", 10))
    viz_cfg = bcfg.get("visualization", {})
    write_contact_sheet = bool(viz_cfg.get("write_contact_sheet", bcfg.get("write_contact_sheet", True)))
    contact_sheet_k = int(viz_cfg.get("max_items", bcfg.get("contact_sheet_k", worst_k)))
    contact_sheet_tile_width = int(viz_cfg.get("tile_width", bcfg.get("contact_sheet_tile_width", 220)))
    gates = quality_gates_from_config(bcfg.get("quality_gates", {}))
    out_dir = resolve_path(root, bcfg.get("output_dir", "runs/coconut_benchmark"))
    report_jsonl = out_dir / "benchmark_records.jsonl"
    summary_md = out_dir / "benchmark_summary.md"
    pred_mask_dir = resolve_path(root, bcfg.get("pred_mask_dir", out_dir / "pred_masks"))
    gt_mask_dir = resolve_path(root, bcfg.get("gt_mask_dir", out_dir / "gt_masks"))
    diff_mask_dir = resolve_path(root, bcfg.get("diff_mask_dir", out_dir / "diff_masks"))

    if dry_run:
        log.info("[dry-run] would benchmark up to %d COCONut val images -> %s", limit, out_dir)
        return report_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[BenchmarkRecord] = []
    manifest_rows: list[MediaItem] = []
    ann_rows: list[AnnotationRecord] = []

    t0 = time.perf_counter()
    import cv2

    runtime = labeling_runtime_from_config(cfg, segment_mode=sam_mode)

    for sample in iter_coconut_person_samples(
        json_path=json_path,
        mask_dir=mask_dir,
        image_root=image_root,
        image_subdir=str(bcfg.get("image_subdir", "val2017")),
        limit=limit,
        seed=seed,
    ):
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            continue
        manifest_rows.append(sample_to_media_item(sample))
        instances: list[InstanceAnnotation] = []
        yolo_detections: list[PersonDetection] = []
        yolo_used: set[int] = set()
        if detector_mode == "yolo_person":
            yolo_detections = detect_persons(
                image_bgr,
                weights=yolo_weights,
                conf=yolo_conf,
                iou=yolo_iou,
                device=device,
            )

        for idx, person in enumerate(sample.persons):
            inst_t0 = time.perf_counter()
            gt = person.mask
            det_bbox, det_meta = _detect_person_bbox(
                person,
                sample,
                mode=detector_mode,
                rng=rng,
                yolo_detections=yolo_detections,
                yolo_used=yolo_used,
                yolo_match_iou=yolo_match_iou,
            )
            neighbor_bboxes = [p.bbox_xyxy for j, p in enumerate(sample.persons) if j != idx]
            result = label_instance_from_bbox(
                image_bgr,
                bbox_xyxy=det_bbox,
                width=sample.width,
                height=sample.height,
                runtime=runtime,
                cfg=cfg,
                gt_mask=gt,
                multi_person=len(sample.persons) > 1,
                detector_meta=det_meta,
                neighbor_bboxes=neighbor_bboxes,
            )
            pred = result.mask
            prompt = result.prompt
            elapsed_ms = (time.perf_counter() - inst_t0) * 1000.0
            iou = float(result.quality_scores.get("semantic_score", 0.0))
            bf1 = float(result.quality_scores.get("boundary_score", 0.0))
            det_bbox_iou = bbox_iou(det_bbox, person.bbox_xyxy)
            stats = {
                k: float(v)
                for k, v in result.quality_scores.items()
                if k in {"gt_area_ratio", "pred_area_ratio", "false_positive_ratio", "false_negative_ratio"}
            }
            decision = result.decision
            error_tags = result.error_tags
            improvement_hint = result.improvement_hint
            mask_stem = f"{sample.image_path.stem}_person_{person.instance_index}"
            pred_mask_path = pred_mask_dir / f"{mask_stem}_pred.png"
            gt_mask_path = gt_mask_dir / f"{mask_stem}_gt.png"
            diff_mask_path = diff_mask_dir / f"{mask_stem}_diff.png"
            if write_masks:
                write_binary_mask(pred_mask_path, pred)
                write_binary_mask(gt_mask_path, gt)
                write_uint8_image(diff_mask_path, _diff_mask(pred, gt))
            else:
                pred_mask_path = Path("")
                gt_mask_path = sample.mask_path
                diff_mask_path = Path("")
            seg_source = result.segment_source
            quality_scores = {
                **result.quality_scores,
                "bbox_iou": float(det_bbox_iou),
            }
            instances.append(
                InstanceAnnotation(
                    instance_id=f"person_{person.instance_index}",
                    bbox_xyxy=mask_to_bbox_xyxy(pred) or det_bbox,
                    mask_path=str(pred_mask_path) if write_masks else None,
                    score=float(prompt.confidence),
                    source=seg_source,
                    target_id=f"seg_{person.segment_id}",
                    prompt_history=[
                        {
                            "agent": prompt.policy,
                            "prompts": list(prompt.prompts),
                            "decision": decision,
                            "error_tags": error_tags,
                            "improvement_hint": improvement_hint,
                        }
                    ],
                )
            )
            records.append(
                BenchmarkRecord(
                    item_id=sample.image_path.stem,
                    instance_id=f"person_{person.instance_index}",
                    image_path=str(sample.image_path),
                    gt_mask_path=str(gt_mask_path),
                    pred_mask_path=str(pred_mask_path) if write_masks else "",
                    diff_mask_path=str(diff_mask_path) if write_masks else "",
                    detector_mode=detector_mode,
                    sam_mode=sam_mode,
                    mask_iou=float(iou),
                    boundary_f_score=float(bf1),
                    bbox_iou=float(det_bbox_iou),
                    gt_area_ratio=stats["gt_area_ratio"],
                    pred_area_ratio=stats["pred_area_ratio"],
                    false_positive_ratio=stats["false_positive_ratio"],
                    false_negative_ratio=stats["false_negative_ratio"],
                    prompt_confidence=float(prompt.confidence),
                    needs_scribble=bool(prompt.needs_scribble),
                    decision=decision,  # type: ignore[arg-type]
                    error_tags=error_tags,
                    improvement_hint=improvement_hint,
                    elapsed_ms=float(elapsed_ms),
                    quality_scores=quality_scores,
                )
            )
        ann_rows.append(AnnotationRecord(item_id=sample.image_path.stem, instances=instances))

    total_s = time.perf_counter() - t0
    write_jsonl(report_jsonl, records, overwrite=True)
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows, overwrite=True)
    write_jsonl(out_dir / "annotations_pred.jsonl", ann_rows, overwrite=True)

    summary = _build_summary(
        records=records,
        dataset="coconut/relabeled_coco_val",
        detector_mode=detector_mode,
        sam_mode=sam_mode,
        total_s=total_s,
        quality_gates=gates,
        worst_k=worst_k,
    )
    if write_contact_sheet:
        summary["contact_sheet_path"] = str(
            write_benchmark_contact_sheet(
                out_dir,
                records=records,
                max_items=contact_sheet_k,
                tile_width=contact_sheet_tile_width,
            )
        )
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    log.info(
        "COCONut benchmark: n=%d IoU=%.4f BF1=%.4f accept=%.1f%% throughput=%.2f inst/s -> %s",
        len(records),
        float(summary["mean_mask_iou"]),
        float(summary["mean_boundary_f1"]),
        100.0 * float(summary["accept_rate"]),
        float(summary["instances_per_second"]),
        out_dir,
    )
    return report_jsonl


def _record_needs_backfill(record: BenchmarkRecord) -> bool:
    return not record.error_tags and record.decision == "review" and record.false_positive_ratio is None


def backfill_benchmark_records(
    records: list[BenchmarkRecord],
    *,
    gates: dict[str, float] | None = None,
) -> list[BenchmarkRecord]:
    """Recompute QA fields for legacy benchmark JSONL rows."""
    gates = quality_gates_from_config(gates)
    updated: list[BenchmarkRecord] = []
    for record in records:
        if not _record_needs_backfill(record):
            updated.append(record)
            continue
        iou = float(record.mask_iou)
        bf1 = float(record.boundary_f_score)
        stats = {
            "gt_area_ratio": record.gt_area_ratio or 0.0,
            "pred_area_ratio": record.pred_area_ratio or 0.0,
            "false_positive_ratio": record.false_positive_ratio or max(0.0, 1.0 - iou),
            "false_negative_ratio": record.false_negative_ratio or max(0.0, 1.0 - iou),
        }
        decision, error_tags, hint = _decision_and_tags(
            iou=iou,
            boundary=bf1,
            stats=stats,
            gates=gates,
            prompt_needs_scribble=bool(record.needs_scribble),
            detector_meta={
                k: float(v)
                for k, v in record.quality_scores.items()
                if k in {"det_matched", "det_match_iou", "det_score"}
            },
            multi_person="multi_person" in record.error_tags or record.quality_scores.get("multi_person", 0) > 0,
            pred_empty=iou <= 0.0 and bf1 <= 0.0,
        )
        updated.append(
            record.model_copy(
                update={
                    "decision": decision,
                    "error_tags": error_tags,
                    "improvement_hint": hint,
                    "false_positive_ratio": stats["false_positive_ratio"],
                    "false_negative_ratio": stats["false_negative_ratio"],
                }
            )
        )
    return updated


def resummarize_benchmark_dir(
    out_dir: Path,
    *,
    quality_gates: dict[str, float] | None = None,
    worst_k: int = 10,
    rewrite_records: bool = True,
    write_contact_sheet: bool = True,
) -> dict[str, object]:
    """Rebuild summary artifacts from an existing benchmark directory."""
    report_jsonl = out_dir / "benchmark_records.jsonl"
    records = read_jsonl_list(report_jsonl, model=BenchmarkRecord)
    if not records:
        raise FileNotFoundError(f"no benchmark records in {report_jsonl}")
    gates = quality_gates_from_config(quality_gates)
    records = backfill_benchmark_records(records, gates=gates)
    if rewrite_records:
        write_jsonl(report_jsonl, records, overwrite=True)
    detector_mode = records[0].detector_mode
    sam_mode = records[0].sam_mode
    total_s = _mean([r.elapsed_ms for r in records]) * len(records) / 1000.0
    summary = _build_summary(
        records=records,
        dataset="coconut/relabeled_coco_val",
        detector_mode=detector_mode,
        sam_mode=sam_mode,
        total_s=total_s,
        quality_gates=gates,
        worst_k=worst_k,
    )
    if write_contact_sheet:
        summary["contact_sheet_path"] = str(write_benchmark_contact_sheet(out_dir, records=records, max_items=worst_k))
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "benchmark_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def export_benchmark_review_queue(
    out_dir: Path,
    *,
    review_path: Path | None = None,
    decisions: tuple[str, ...] = ("review", "reject"),
) -> Path:
    """Export review/reject benchmark instances for active labeling."""
    report_jsonl = out_dir / "benchmark_records.jsonl"
    records = read_jsonl_list(report_jsonl, model=BenchmarkRecord)
    gates = quality_gates_from_config()
    records = backfill_benchmark_records(records, gates=gates)
    review_path = review_path or (out_dir / "review_queue.jsonl")
    rows = [
        {
            "item_id": r.item_id,
            "instance_id": r.instance_id,
            "image_path": r.image_path,
            "gt_mask_path": r.gt_mask_path,
            "pred_mask_path": r.pred_mask_path,
            "diff_mask_path": r.diff_mask_path,
            "decision": r.decision,
            "mask_iou": r.mask_iou,
            "boundary_f_score": r.boundary_f_score,
            "error_tags": r.error_tags,
            "improvement_hint": r.improvement_hint,
            "quality_scores": r.quality_scores,
            "suggested_actions": ["prompt_correction", "SAM2_repropagation", "boundary_paint"],
        }
        for r in records
        if r.decision in decisions
    ]
    write_jsonl(review_path, rows, overwrite=True)
    log.info("Exported %d benchmark review items -> %s", len(rows), review_path)
    return review_path
