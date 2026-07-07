"""Grounded-SAM-2 external adapter — open-vocabulary person discovery.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``grounded_sam2``.
The default command template invokes ``python -m grounded_sam2.detect``
from a standard IDEA-Research/Grounded-SAM-2 checkout in a GPU env.

Outputs (each a JSON file written by the CLI):
``person_candidates`` (candidate list), ``bbox`` (per-candidate boxes),
``rle_mask`` (per-candidate RLE masks), ``score`` (per-candidate scores).
Grounded-SAM-2 is the priority-1 human-discovery adapter; its candidates
bootstrap SAM2 temporal masklet propagation (A2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["GroundedSam2Adapter"]

INTEGRATION = "grounded_sam2"

OUTPUT_FILES = {
    "person_candidates": "person_candidates.json",
    "bbox": "bbox.json",
    "rle_mask": "rle_mask.json",
    "score": "score.json",
}


class GroundedSam2Adapter(SubprocessAdapter):
    """Typed adapter for external Grounded-SAM-2 person discovery."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        text_prompt: str = "person",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 600.0,
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
        self.text_prompt = text_prompt

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def detect(
        self,
        image: str | Path,
        *,
        output_dir: str | Path,
        text_prompt: Optional[str] = None,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Open-vocab person discovery on one image; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image)}
        params = {"repo_python": self.repo_python, "text_prompt": text_prompt or self.text_prompt}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs