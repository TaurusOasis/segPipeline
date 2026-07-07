"""Mask-refinement external adapters (pipeline step 5 / step 12).

Concrete :class:`~hmp.adapters.base.SubprocessAdapter` subclasses that turn
the generic contract into a typed, mask-refine-shaped API:

* :class:`SamRefinerAdapter` — coarse-mask boundary refinement via the
  external SAMRefiner repo (``refined_mask`` + ``mask_quality`` outputs).
* :class:`HqSamAdapter` — SAM-HQ boundary refinement from a box prompt
  (``refined_mask`` output).
* :class:`CascadePSPAdapter` — high-resolution mask refinement (priority-3,
  ``refined_mask`` output).
"""

from __future__ import annotations

from .cascadepsp import CascadePSPAdapter
from .hq_sam import HqSamAdapter
from .samrefiner import SamRefinerAdapter

__all__ = ["SamRefinerAdapter", "HqSamAdapter", "CascadePSPAdapter"]