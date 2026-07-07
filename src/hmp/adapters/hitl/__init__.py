"""Human-in-the-loop (HITL) external adapters (pipeline step 10).

External annotation/correction tools that turn model outputs into
human-verified edits. These are subprocess adapters like the model
adapters, but their command templates launch a tool bridge (or a
non-blocking view) rather than a model inference.

* :class:`FiftyoneAdapter` — FiftyOne dataset view + reviewer selection
  export (``dataset_view`` + ``review_selection`` outputs).
* :class:`CvatAdapter` — CVAT task correction bridge
  (``human_edits`` + ``corrected_prompts`` + ``audit_log`` outputs).
* :class:`LabelStudioAdapter` — Label Studio project correction bridge
  (``human_edits`` + ``audit_log`` outputs).
"""

from __future__ import annotations

from .cvat import CvatAdapter
from .fiftyone import FiftyoneAdapter
from .label_studio import LabelStudioAdapter

__all__ = ["FiftyoneAdapter", "CvatAdapter", "LabelStudioAdapter"]