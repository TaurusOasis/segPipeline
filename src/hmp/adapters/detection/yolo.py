"""Ultralytics YOLO external adapter — high-throughput person detection baseline.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``ultralytics_yolo``.
The default command template invokes ``python -m hmp.yolo.detect_cli`` (the
pipeline's own thin YOLO inference CLI) so this adapter reuses the trained
edge-student weights as a discovery/detection baseline in the GPU env.

Outputs (each a JSON file): ``bbox``, ``mask``, ``score``. Ultralytics YOLO
is the priority-2 human-discovery adapter; AGPL — see ``license_review``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["YoloDetectAdapter"]

INTEGRATION = "ultralytics_yolo"

OUTPUT_FILES = {
    "bbox": "bbox.json",
    "mask": "mask.json",
    "score": "score.json",
}


class YoloDetectAdapter(SubprocessAdapter):
    """Typed adapter for Ultralytics YOLO detection/segmentation baseline."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        weights: str = "yolo26s-seg.pt",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 300.0,
        command_template: Optional[list[str]] = None,
        registry: Optional[AdapterRegistry] = None,
    ) -> None:
        reg = registry or load_registry()
        spec = reg.get(INTEGRATION)
        tmpl = list(command_template) if command_template is not None else template_for(INTEGRATION)
        env_overlay = dict(env or {})
        env_overlay.setdefault("REPO_PYTHON", repo_python)
        super().__init__(
            spec,
            workdir=workdir,
            command_template=tmpl,
            env=env_overlay,
            timeout_s=timeout_s,
        )
        self.repo_python = repo_python
        self.weights = weights

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def detect(
        self,
        image: str | Path,
        *,
        output_dir: str | Path,
        weights: Optional[str] = None,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """YOLO detect/segment one image; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image)}
        params = {"repo_python": self.repo_python, "weights": weights or self.weights}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs