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
from .templates import (
    ADAPTER_INPUT_KEYS,
    ADAPTER_OUTPUT_KEYS,
    DEFAULT_COMMAND_TEMPLATES,
    build_adapter,
    dry_run_adapter,
    template_for,
)


def _build_concrete_registry() -> "dict[str, type]":
    """Build the name -> concrete adapter class map (import-side-effect free)."""
    from .active_labeling import GymnasiumAdapter, StableBaselines3Adapter
    from .detection import GroundedSam2Adapter, GroundingDinoAdapter, YoloDetectAdapter
    from .diffusion import DiffMatteAdapter, SDMatteAdapter, VideoMaMaAdapter
    from .hitl import CvatAdapter, FiftyoneAdapter, LabelStudioAdapter
    from .mask_refine import CascadePSPAdapter, HqSamAdapter, SamRefinerAdapter
    from .matting import (
        MaggieAdapter,
        MatAnyone2Adapter,
        MatAnyoneAdapter,
        MattingAnythingAdapter,
        RvmAdapter,
        SematAdapter,
    )
    from .qa import MmagicMetricsAdapter, RaftAdapter
    from .vos import CutieAdapter, XMemAdapter

    return {
        # mask_refine
        "samrefiner": SamRefinerAdapter,
        "hq_sam": HqSamAdapter,
        "cascadepsp": CascadePSPAdapter,
        # matting
        "matanyone": MatAnyoneAdapter,
        "matanyone2": MatAnyone2Adapter,
        "semat": SematAdapter,
        "matting_anything": MattingAnythingAdapter,
        "maggie": MaggieAdapter,
        "rvm": RvmAdapter,
        # vos
        "cutie": CutieAdapter,
        "xmem": XMemAdapter,
        # diffusion
        "videomama": VideoMaMaAdapter,
        "diffmatte": DiffMatteAdapter,
        "sdmatte": SDMatteAdapter,
        # qa
        "raft": RaftAdapter,
        "mmagic": MmagicMetricsAdapter,
        # detection
        "grounded_sam2": GroundedSam2Adapter,
        "groundingdino": GroundingDinoAdapter,
        "ultralytics_yolo": YoloDetectAdapter,
        # hitl / data_management
        "fiftyone": FiftyoneAdapter,
        "cvat": CvatAdapter,
        "label_studio": LabelStudioAdapter,
        # active_labeling
        "gymnasium": GymnasiumAdapter,
        "stable_baselines3": StableBaselines3Adapter,
    }


#: ``name -> concrete :class:`SubprocessAdapter` subclass`` map for the 25
#: integrations that have a typed adapter. Built eagerly at import (the group
#: modules only define lightweight subclasses, no torch/ultralytics).
CONCRETE_ADAPTERS: dict[str, type] = _build_concrete_registry()


def get_concrete_adapter_class(name: str) -> "type | None":
    """Return the concrete adapter class for ``name`` (None if none exists)."""
    return CONCRETE_ADAPTERS.get(name)


__all__ = [
    "AdapterSpec",
    "ExternalAdapter",
    "SubprocessAdapter",
    "AdapterResult",
    "AdapterRegistry",
    "load_registry",
    "DEFAULT_COMMAND_TEMPLATES",
    "ADAPTER_INPUT_KEYS",
    "ADAPTER_OUTPUT_KEYS",
    "build_adapter",
    "dry_run_adapter",
    "template_for",
    "CONCRETE_ADAPTERS",
    "get_concrete_adapter_class",
]