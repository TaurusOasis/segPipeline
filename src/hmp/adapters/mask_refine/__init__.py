"""Mask-refinement external adapters (pipeline step 5 / step 12).

Concrete :class:`~hmp.adapters.base.SubprocessAdapter` subclasses that turn
the generic contract into a typed, mask-refine-shaped API:

* :class:`SamRefinerAdapter` — coarse-mask boundary refinement via the
  external SAMRefiner repo (``refined_mask`` + ``mask_quality`` outputs).
* :class:`HqSamAdapter` — SAM-HQ boundary refinement from a box prompt
  (``refined_mask`` output).

These wrap the catalog templates from :mod:`hmp.adapters.templates`; the
actual external repo is invoked as a subprocess in a GPU env. The typed
``refine`` / ``refine_batch`` helpers build the input/output maps, call
:meth:`ExternalAdapter.run` (or :meth:`ExternalAdapter.dry_run`), validate
outputs, and emit provenance rows — so the contract is exercised
end-to-end without the repo present (tests use a mock command that writes
the output file).
"""

from __future__ import annotations

from .hq_sam import HqSamAdapter
from .samrefiner import SamRefinerAdapter

__all__ = ["SamRefinerAdapter", "HqSamAdapter"]