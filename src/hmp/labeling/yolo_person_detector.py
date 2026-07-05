"""YOLO person detector adapter (pipeline step 2, lazy ultralytics import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonDetection:
    bbox_xyxy: list[int]
    score: float
    class_id: int = PERSON_CLASS_ID


def bbox_iou(a: list[int], b: list[int]) -> float:
    """Axis-aligned bbox IoU for ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    nx1 = int(max(0, min(width - 1, round(x1))))
    ny1 = int(max(0, min(height - 1, round(y1))))
    nx2 = int(max(nx1 + 1, min(width, round(x2))))
    ny2 = int(max(ny1 + 1, min(height, round(y2))))
    return [nx1, ny1, nx2, ny2]


def detect_persons(
    image_bgr: np.ndarray,
    *,
    weights: str,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | int = "",
    imgsz: int = 640,
) -> list[PersonDetection]:
    """Run Ultralytics YOLO and return COCO-person detections only."""
    from ultralytics import YOLO

    h, w = image_bgr.shape[:2]
    model = YOLO(weights)
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        device=device,
        imgsz=imgsz,
        verbose=False,
    )
    detections: list[PersonDetection] = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for box, score, cls_id in zip(xyxy, scores, classes):
        if int(cls_id) != PERSON_CLASS_ID:
            continue
        detections.append(
            PersonDetection(
                bbox_xyxy=_clip_bbox(list(box), w, h),
                score=float(score),
                class_id=int(cls_id),
            )
        )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def match_detection_for_gt(
    detections: list[PersonDetection],
    gt_bbox: list[int],
    *,
    used_indices: set[int],
    iou_threshold: float = 0.3,
) -> tuple[Optional[PersonDetection], float]:
    """Return the best unused detection for one GT bbox and its IoU."""
    best_idx = -1
    best_iou = 0.0
    for idx, det in enumerate(detections):
        if idx in used_indices:
            continue
        iou = bbox_iou(det.bbox_xyxy, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    if best_idx < 0 or best_iou < iou_threshold:
        return None, best_iou
    used_indices.add(best_idx)
    return detections[best_idx], best_iou
