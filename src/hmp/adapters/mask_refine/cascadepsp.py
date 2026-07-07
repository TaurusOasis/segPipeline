"""CascadePSP external adapter — high-resolution mask refinement.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``cascadepsp``. The
default command template invokes ``python -m cascadepsp.refine`` from a
standard hkchengrex/CascadePSP checkout in a GPU env.

Output: ``refined_mask`` (the high-resolution refined mask PNG). CascadePSP
is the priority-3 mask refiner behind SAMRefiner (p1) and SamHQ (p2), used for
global high-resolution refinement when the coarse mask is low-res.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["CascadePSPAdapter"]

INTEGRATION = "cascadepsp"


class CascadePSPAdapter(SubprocessAdapter):
    """Typed adapter for external CascadePSP high-resolution mask refinement."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
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

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {"refined_mask": out / "refined_mask.png"}

    def refine(
        self,
        image: str | Path,
        coarse_mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """High-res refine a coarse mask; return (result, output_paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image), "coarse_mask": str(coarse_mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs