"""MMagic external adapter — matting quality metrics (SAD/MSE/gradient/connectivity).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``mmagic``. The
default command template invokes ``python -m mmagic.matting_metrics`` from
an open-mmlab/mmagic checkout, computing the four standard matting-quality
metrics between predicted alphas (pred-dir), ground-truth alphas (gt-dir),
and optional trimaps (trimap-dir).

Outputs (each a JSON file containing the scalar metric): ``sad``,
``mse``, ``gradient``, ``connectivity``. MMagic metrics is the priority-2
QA adapter, a reference implementation of the same metrics the pipeline's
own :mod:`hmp.eval.alpha_metrics` produces — used to cross-check the
internal evaluator on benchmark splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["MmagicMetricsAdapter"]

INTEGRATION = "mmagic"

OUTPUT_FILES = {
    "sad": "sad.json",
    "mse": "mse.json",
    "gradient": "gradient.json",
    "connectivity": "connectivity.json",
}


class MmagicMetricsAdapter(SubprocessAdapter):
    """Typed adapter for external MMagic matting-metric computation."""

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
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def metrics(
        self,
        pred_dir: str | Path,
        gt_dir: str | Path,
        trimap_dir: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Compute SAD/MSE/gradient/connectivity; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            "trimap_dir": str(trimap_dir),
        }
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs