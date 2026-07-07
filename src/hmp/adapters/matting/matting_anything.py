"""Matting-Anything (MAM) external adapter — SAM-feature mask-to-alpha (Bi).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``matting_anything``.
The default command template invokes ``python -m mam.infer`` from a
standard shi-labs/Matting-Anything checkout in a GPU env.

Output: ``alpha_image`` (the image alpha matte PNG). MAM uses SAM features
plus a mask-to-matte head, an alternative Bi branch to SEMat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["MattingAnythingAdapter"]

INTEGRATION = "matting_anything"


class MattingAnythingAdapter(SubprocessAdapter):
    """Typed adapter for external Matting-Anything image matting (Bi alt)."""

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
        return {"alpha_image": out / "alpha_image.png"}

    def mat(
        self,
        image: str | Path,
        mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Run SAM-feature mask-to-alpha on one image; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image), "mask": str(mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs