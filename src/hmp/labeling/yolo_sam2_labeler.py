"""YOLO edge detector + GPU segment teacher (SAM2/SamHQ/GrabCut) for steps 2-4."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..common.logging import get_logger
from ..data.mask_io import mask_to_bbox_xyxy, write_binary_mask
from ..schemas import InstanceAnnotation, MediaItem
from .auto_label_core import detect_persons_for_image, label_instance_from_bbox, labeling_runtime_from_config
from ..models.tiers import load_model_tiers
from .base import Labeler
from .sam_teacher import teacher_segment_source

log = get_logger("hmp.labeling.yolo_sam2")

SegmentMode = Literal["grabcut", "sam2", "samhq"]


class YoloSam2Labeler(Labeler):
    """Detect persons with edge YOLO; segment masks with GPU teacher (SAM2/SamHQ)."""

    name = "yolo_sam2"

    def __init__(
        self,
        cfg,
        *,
        project_root: Optional[Path] = None,
        segment_mode: SegmentMode | None = None,
        teacher_key: str | None = None,
        **kw,
    ) -> None:
        super().__init__(cfg, project_root=project_root, **kw)
        self.runtime = labeling_runtime_from_config(
            cfg,
            segment_mode=segment_mode,
            teacher_key=teacher_key,
        )
        self.segment_mode = self.runtime.segment_mode
        self.teacher_key = self.runtime.teacher_key
        backend = self.runtime.teacher.backend if self.runtime.teacher else self.segment_mode
        self.name = f"yolo_{teacher_segment_source(backend)}"  # type: ignore[arg-type]

    def _read_image_bgr(self, item: MediaItem):
        import cv2

        image = cv2.imread(str(item.path))
        if image is None:
            raise FileNotFoundError(f"failed to read image: {item.path}")
        return image

    def label_one(self, item: MediaItem) -> list[InstanceAnnotation]:
        image_bgr = self._read_image_bgr(item)
        detections = detect_persons_for_image(image_bgr, self.runtime)

        instances: list[InstanceAnnotation] = []
        for idx, det in enumerate(detections):
            neighbor_bboxes = [d.bbox_xyxy for j, d in enumerate(detections) if j != idx]
            result = label_instance_from_bbox(
                image_bgr,
                bbox_xyxy=det.bbox_xyxy,
                width=item.width,
                height=item.height,
                runtime=self.runtime,
                cfg=self.cfg,
                det_score=float(det.score),
                multi_person=len(detections) > 1,
                detector_meta={"det_score": float(det.score), "det_matched": 1.0},
                neighbor_bboxes=neighbor_bboxes,
            )
            mask_path = self.mask_dir / f"{item.item_id}_person_{idx}.png"
            write_binary_mask(mask_path, result.mask)
            bbox = mask_to_bbox_xyxy(result.mask) or det.bbox_xyxy
            instances.append(
                InstanceAnnotation(
                    instance_id=f"person_{idx}",
                    category="person",
                    bbox_xyxy=bbox,
                    mask_path=str(mask_path),
                    score=float(det.score),
                    source=f"yolo+{result.segment_source}",
                    prompt_history=[
                        {
                            "agent": result.prompt.policy,
                            "prompts": list(result.prompt.prompts),
                            "decision": result.decision,
                            "error_tags": result.error_tags,
                            "improvement_hint": result.improvement_hint,
                            "quality_scores": result.quality_scores,
                            "edge_model": load_model_tiers(self.cfg).edge.name,
                            "teacher_model": self.teacher_key,
                        }
                    ],
                )
            )
        if not instances:
            log.warning("[%s] no person detections for %s", self.name, item.item_id)
        return instances
