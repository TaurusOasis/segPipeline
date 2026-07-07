"""Diffusion-refine external adapters (pipeline step 7, branch Bd).

Diffusion-based matte refinement runs only inside an adaptive ROI to fix
complex boundaries and motion blur that the mask/alpha teachers got
partially right. They are the most expensive branch and run last.

* :class:`VideoMaMaAdapter` — video diffusion mask-to-matte branch Bd
  (``alpha_diffusion`` + ``refine_roi`` outputs).
* :class:`DiffMatteAdapter` — image/keyframe diffusion matting refine
  (``alpha_diffusion`` output).
* :class:`SDMatteAdapter` — Stable-Diffusion-based interactive image
  matting reference (``alpha_diffusion`` output).
"""

from __future__ import annotations

from .diffmatte import DiffMatteAdapter
from .sdmatte import SDMatteAdapter
from .videomama import VideoMaMaAdapter

__all__ = ["VideoMaMaAdapter", "DiffMatteAdapter", "SDMatteAdapter"]