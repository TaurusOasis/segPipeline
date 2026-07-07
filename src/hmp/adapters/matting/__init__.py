"""Matting-stage external adapters (pipeline step 7, alpha teacher branches).

* :class:`MatAnyoneAdapter` — target-assigned human video matting branch Bv
  (``alpha_video`` + ``branch_source`` outputs).
"""

from __future__ import annotations

from .matanyone import MatAnyoneAdapter

__all__ = ["MatAnyoneAdapter"]