"""MatAnyone2 external adapter — real-video matting with eval_map (branch Bv).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``matanyone2``. The
default command template invokes ``python -m matanyone2.infer`` from a
standard pq-yang/MatAnyone2 checkout in a GPU env.

Outputs: ``alpha_video`` (the matte video), ``eval_map`` (per-pixel
reliability/error map directory the MQE stage reads), and ``quality_score``
(a JSON clip quality score). MatAnyone2 is the real-video matting reference
used to design the MQE, so its eval_map output is the reason it's distinct
from MatAnyone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["MatAnyone2Adapter"]

INTEGRATION = "matanyone2"


class MatAnyone2Adapter(SubprocessAdapter):
    """Typed adapter for external MatAnyone2 real-video matting (branch Bv)."""

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
            "alpha_video": out / "alpha.mp4",
            "eval_map": out / "eval_map",
            "quality_score": out / "quality_score.json",
        }

    def mat(
        self,
        image_dir: str | Path,
        target_mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
        target_id: Optional[str] = None,
    ) -> "tuple[Any, dict[str, Path]]":
        """Run real-video matting with eval_map; return (result, output_paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image_dir": str(image_dir), "target_mask": str(target_mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs