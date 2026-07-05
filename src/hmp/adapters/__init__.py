"""External research-repo adapter contracts (offline GPU data engine layer).

Per ``doc/CODE_TARGETS_MEM_zh.md`` and ``configs/reference_integrations.yaml``,
external projects (SAM2, SamHQ, SAMRefiner, Cutie/XMem, RAFT/GMFlow,
MatAnyone, MaGGIe, SEMat, VideoMaMa, CVAT, ...) are integrated **only** through
adapter / subprocess / command-template boundaries — never by vendoring their
source into ``src/``. Every adapter returns outputs that flow back into hmp's
JSONL schema, QA, and review queue.

This module defines the base contract shared by all those adapters:

* :class:`AdapterSpec`        — declarative metadata loaded from
  ``configs/reference_integrations.yaml`` (group, url, expected_outputs,
  license_review, adapter_target).
* :class:`ExternalAdapter`    — abstract base: command template, env overlay,
  dry-run, real run, output validation, provenance emission.
* :class:`SubprocessAdapter`  — concrete generic subprocess adapter (the
  default for any integration whose command is a plain argv template).
* :class:`AdapterResult`      — structured run outcome (returncode, stdout,
  stderr, duration, resolved outputs, dry_run flag).
* :class:`AdapterRegistry`    — loads specs from the yaml registry and looks
  them up by integration name.

The contract is deliberately CPU-only and stdlib+yaml: no torch, no
ultralytics, no GPU. Heavy adapters subclass :class:`ExternalAdapter` and
lazy-import their deps inside :meth:`run`.
"""

from __future__ import annotations

from .base import (
    AdapterRegistry,
    AdapterResult,
    AdapterSpec,
    ExternalAdapter,
    SubprocessAdapter,
    load_registry,
)

__all__ = [
    "AdapterSpec",
    "ExternalAdapter",
    "SubprocessAdapter",
    "AdapterResult",
    "AdapterRegistry",
    "load_registry",
]