"""DiffMatte diffusion adapter — image/keyframe matting refine (branch Bd, image).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``diffmatte``. The
default command template invokes ``python -m diffmatte.infer`` from a
standard YihanHu-2022/DiffMatte checkout in a GPU env.

Output: ``alpha_diffusion`` (the refined image/keyframe alpha matte PNG).
Used to refine a single hard-boundary frame (e.g. a keyframe) before the
video matte branch runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["DiffMatteAdapter"]

INTEGRATION = "diffmatte"


class DiffMatteAdapter(SubprocessAdapter):
    """Typed adapter for external DiffMatte image/keyframe diffusion refine."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 1200.0,
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
        return {"alpha_diffusion": out / "alpha_diffusion.png"}

    def refine(
        self,
        image: str | Path,
        mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Diffusion-refine one image/keyframe alpha; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image), "mask": str(mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs