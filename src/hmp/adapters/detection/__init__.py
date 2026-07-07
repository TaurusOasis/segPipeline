"""Detection / human-discovery external adapters (pipeline step 1).

Open-vocabulary and trained detectors that produce person candidates
(bboxes, masks, scores) for the prompt agent and downstream matting.

* :class:`GroundedSam2Adapter` — Grounded-SAM-2 person discovery plus SAM2
  tracking bootstrap (priority-1; ``person_candidates`` + ``bbox`` +
  ``rle_mask`` + ``score`` outputs).
* :class:`GroundingDinoAdapter` — GroundingDINO text-prompt detection
  (priority-2; ``bbox`` + ``score`` + ``phrase`` outputs).
* :class:`YoloDetectAdapter` — Ultralytics YOLO high-throughput
  detector/segmenter baseline (priority-2; ``bbox`` + ``mask`` + ``score``
  outputs).
"""

from __future__ import annotations

from .grounded_sam2 import GroundedSam2Adapter
from .groundingdino import GroundingDinoAdapter
from .yolo import YoloDetectAdapter

__all__ = ["GroundedSam2Adapter", "GroundingDinoAdapter", "YoloDetectAdapter"]