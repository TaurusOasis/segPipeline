"""YOLO person detection + prompt agent + SAM2/GrabCut labeling (steps 2-4)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..agents.prompt_agent import plan_prompts
from ..common.logging import get_logger
from ..data.mask_io import mask_to_bbox_xyxy, write_binary_mask
from ..eval.label_quality import decision_and_tags
from ..refine.mask_postprocess import postprocess_from_config
from ..schemas import InstanceAnnotation, MediaItem
from .auto_label_core import labeling_runtime_from_config
from .base import Labeler
from .mock_sam2 import segment_with_prompts
from .sam2_adapter import segment_with_sam2
from .yolo_person_detector import detect_persons

log = get_logger("hmp.labeling.yolo_sam2")

SegmentMode = Literal["grabcut", "sam2"]


class YoloSam2Labeler(Labeler):
    """Detect persons with YOLO, plan prompts, segment with SAM2 or GrabCut."""

    name = "yolo_sam2"

    def __init__(self, cfg, *, project_root: Optional[Path] = None, segment_mode: SegmentMode | None = None, **kw) -> None:
        super().__init__(cfg, project_root=project_root, **kw)
        self.runtime = labeling_runtime_from_config(cfg, segment_mode=segment_mode)
        self.segment_mode = self.runtime.segment_mode

    def _read_image_bgr(self, item: MediaItem) -> np.ndarray:
        import cv2

        image = cv2.imread(str(item.path))
        if image is None:
            raise FileNotFoundError(f"failed to read image: {item.path}")
        return image

    def label_one(self, item: MediaItem) -> list[InstanceAnnotation]:
        image_bgr = self._read_image_bgr(item)
        detections = detect_persons(
            image_bgr,
            weights=self.runtime.yolo_weights,
            conf=self.runtime.yolo_conf,
            iou=self.runtime.yolo_iou,
            device=self.runtime.device,
        )[: self.runtime.max_instances]

        instances: list[InstanceAnnotation] = []
        for idx, det in enumerate(detections):
            prompt = plan_prompts(
                bbox_xyxy=det.bbox_xyxy,
                width=item.width,
                height=item.height,
            )
            if self.runtime.segment_mode == "sam2":
                mask = segment_with_sam2(
                    image_bgr,
                    prompt,
                    weights=self.runtime.sam2_weights,
                    device=self.runtime.device,
                    fallback_grabcut=True,
                )
            else:
                mask = segment_with_prompts(image_bgr, prompt)
            mask = postprocess_from_config(mask, self.cfg)
            decision, tags, hint = decision_and_tags(
                iou=None,
                boundary=None,
                stats={},
                gates=self.runtime.quality_gates,
                prompt_needs_scribble=prompt.needs_scribble,
                detector_meta={"det_score": float(det.score)},
                pred_empty=not bool(np.asarray(mask).any()),
                prompt_confidence=float(prompt.confidence),
            )
            quality_scores = {
                "semantic_score": float(prompt.confidence),
                "boundary_score": float(prompt.confidence),
                "identity_score": float(prompt.confidence),
                "det_score": float(det.score),
            }
            mask_path = self.mask_dir / f"{item.item_id}_person_{idx}.png"
            write_binary_mask(mask_path, mask)
            bbox = mask_to_bbox_xyxy(mask) or det.bbox_xyxy
            instances.append(
                InstanceAnnotation(
                    instance_id=f"person_{idx}",
                    category="person",
                    bbox_xyxy=bbox,
                    mask_path=str(mask_path),
                    score=float(det.score),
                    source=f"yolo+{self.runtime.segment_mode}",
                    prompt_history=[
                        {
                            "agent": prompt.policy,
                            "prompts": list(prompt.prompts),
                            "decision": decision,
                            "error_tags": tags,
                            "improvement_hint": hint,
                            "quality_scores": quality_scores,
                        }
                    ],
                )
            )
        if not instances:
            log.warning("[%s] no person detections for %s", self.name, item.item_id)
        return instances
