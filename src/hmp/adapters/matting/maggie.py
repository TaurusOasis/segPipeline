"""MaGGIe external adapter — mask-guided multi-human instance matting.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``maggie``. The default
command template invokes ``python -m maggie.infer`` from a standard
hmchuong/MaGGIe checkout in a GPU env.

Outputs: ``alpha`` (the composited alpha matte PNG) and ``instance_alpha``
(per-instance alpha). MaGGIe is mask-guided, so the input is an instance mask
that selects which human(s) to matte — used for multi-person frames where the
masklet already separated identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["MaggieAdapter"]

INTEGRATION = "maggie"


class MaggieAdapter(SubprocessAdapter):
    """Typed adapter for external MaGGIe mask-guided matting."""

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
        return {
            "alpha": out / "alpha.png",
            "instance_alpha": out / "instance_alpha.png",
        }

    def mat(
        self,
        image: str | Path,
        instance_mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Run mask-guided matting; return (result, output_paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image), "instance_mask": str(instance_mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs