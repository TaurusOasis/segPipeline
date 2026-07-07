"""Matting-stage external adapters (pipeline step 7, alpha teacher branches).

* :class:`MatAnyoneAdapter` — target-assigned human video matting branch Bv
  (``alpha_video`` + ``branch_source`` outputs).
* :class:`SematAdapter` — semantic/person mask-to-alpha image branch Bi
  (``alpha_image`` output).
* :class:`MaggieAdapter` — mask-guided multi-human instance matting
  (``alpha`` + ``instance_alpha`` outputs).
* :class:`RvmAdapter` — RobustVideoMatting real-time video matting
  baseline/teacher (``alpha_video`` output).
"""

from __future__ import annotations

from .maggie import MaggieAdapter
from .matanyone import MatAnyoneAdapter
from .rvm import RvmAdapter
from .semat import SematAdapter

__all__ = ["MatAnyoneAdapter", "SematAdapter", "MaggieAdapter", "RvmAdapter"]