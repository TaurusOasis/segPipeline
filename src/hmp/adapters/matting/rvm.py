"""RobustVideoMatting external adapter — real-time video matting baseline/teacher.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``rvm``. The default
command template invokes ``python -m rvm.infer`` from a standard
PeterL1n/RobustVideoMatting checkout in a GPU env.

Output: ``alpha_video`` (the matte video). RVM is a background-agnostic
real-time matting baseline used both as a teacher and as a Bv-branch
sanity check against the target-assigned MatAnyone branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["RvmAdapter"]

INTEGRATION = "rvm"


class RvmAdapter(SubprocessAdapter):
    """Typed adapter for external RobustVideoMatting (video matting teacher)."""

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
        return {"alpha_video": out / "alpha.mp4"}

    def mat(
        self,
        image_dir: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Run video matting over a frame directory; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image_dir": str(image_dir)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs