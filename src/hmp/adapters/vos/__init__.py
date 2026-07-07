"""VOS (video object segmentation) masklet adapters (pipeline step 4).

* :class:`CutieAdapter` — Cutie VOS fallback / interactive masklet
  propagation (``masklet`` + ``track_id`` outputs).
* :class:`XMemAdapter` — XMem long-video memory-based VOS fallback
  (``masklet`` + ``track_id`` outputs).

Both take an image directory plus a mask prompt (keyframe mask / scribble
JSON, depending on the repo) and produce a per-frame masklet directory plus a
``track_id.json`` recording the track identity and re-entry metadata. They
are the VOS fallbacks behind SAM2 (priority-1 masklet in the registry), used
when SAM2 masklet propagation (A2) is unavailable or unstable.
"""

from __future__ import annotations

from .cutie import CutieAdapter
from .xmem import XMemAdapter

__all__ = ["CutieAdapter", "XMemAdapter"]