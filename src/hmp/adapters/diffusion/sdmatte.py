"""SDMatte diffusion adapter — Stable-Diffusion-based image matting reference.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``sdmatte``. The
default command template invokes ``python -m sdmatte.infer`` from a
standard vivoCameraResearch/SDMatte checkout in a GPU env.

Output: ``alpha_diffusion`` (the SD-refined image alpha PNG). Lower priority
than DiffMatte; kept as an interactive/reference branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["SDMatteAdapter"]

INTEGRATION = "sdmatte"


class SDMatteAdapter(SubprocessAdapter):
    """Typed adapter for external SDMatte Stable-Diffusion matting reference."""

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
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Refine one image's alpha via Stable Diffusion; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs