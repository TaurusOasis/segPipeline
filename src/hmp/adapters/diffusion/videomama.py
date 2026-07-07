"""VideoMaMa diffusion adapter — video mask-to-matte refine (branch Bd).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``videomama``. The
default command template invokes ``python -m videomama.refine`` from a
standard cvlab-kaist/VideoMaMa checkout in a GPU env.

Outputs: ``alpha_diffusion`` (the refined matte over the ROI) and
``refine_roi`` (a JSON recording the ROI VideoMaMa actually refined, for
provenance and re-running the fusion stage on the right region). This is the
Bd branch — only invoked inside the adaptive ROI on hard boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["VideoMaMaAdapter"]

INTEGRATION = "videomama"


class VideoMaMaAdapter(SubprocessAdapter):
    """Typed adapter for external VideoMaMa diffusion matting (branch Bd)."""

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
        return {
            "alpha_diffusion": out / "alpha_diffusion",
            "refine_roi": out / "refine_roi.json",
        }

    def refine(
        self,
        image_dir: str | Path,
        coarse_alpha_dir: str | Path,
        roi: str,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Diffusion-refine coarse alphas inside ``roi``; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {
            "image_dir": str(image_dir),
            "coarse_alpha_dir": str(coarse_alpha_dir),
            "roi": str(roi),
        }
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs