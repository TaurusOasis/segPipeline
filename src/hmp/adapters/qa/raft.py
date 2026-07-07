"""RAFT optical-flow temporal-consistency QA adapter (pipeline step 8).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``raft``. The default
command template invokes ``python -m raft.compute_flow`` from a standard
princeton-vl/RAFT checkout in a GPU env.

CPU fallback: when the external RAFT repo is not available (the subprocess
returns non-zero, e.g. ``ModuleNotFoundError``) and ``fallback=True`` (the
default), :meth:`run_qa` computes the temporal error from
:mod:`hmp.eval.temporal_metrics` using :func:`frame_diff_flow` (zero flow),
writes the same two output files, and returns the result with
``returncode=0``. This mirrors the roadmap's "RAFT/GMFlow with frame-diff
fallback for CPU smoke tests" and lets the contract validate end-to-end
without the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ...common.logging import get_logger
from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["RaftAdapter"]

log = get_logger("hmp.adapters.qa")

INTEGRATION = "raft"


class RaftAdapter(SubprocessAdapter):
    """Typed adapter for external RAFT temporal-consistency QA, with CPU fallback."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 600.0,
        command_template: Optional[list[str]] = None,
        registry: Optional[AdapterRegistry] = None,
        fallback: bool = True,
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
        self.fallback = fallback

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {
            "temporal_error": out / "temporal_error.json",
            "flow_consistency_score": out / "flow_consistency_score.json",
        }

    def run_qa(
        self,
        prev_alpha: str | Path | np.ndarray,
        cur_alpha: str | Path | np.ndarray,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Compute temporal error between consecutive alphas; return (result, paths).

        ``prev_alpha`` / ``cur_alpha`` may be paths (passed to the external
        CLI) or in-memory arrays (used by the CPU fallback). The two output
        JSON files (``temporal_error.json`` + ``flow_consistency_score.json``)
        are written under ``output_dir``.
        """
        outputs = self._output_paths(output_dir)
        inputs = {
            "prev_alpha": str(prev_alpha),
            "cur_alpha": str(cur_alpha),
        }
        params = {"repo_python": self.repo_python}
        if not execute:
            result = self.dry_run(inputs, outputs, params=params)
            return result, outputs

        result = self.run(inputs, outputs, params=params)
        if not result.ok and self.fallback:
            log.warning(
                "[%s] subprocess failed (rc=%d); using CPU frame-diff fallback",
                self.spec.name, result.returncode,
            )
            self._cpu_fallback(prev_alpha, cur_alpha, outputs, result)
        return result, outputs

    def _cpu_fallback(
        self,
        prev_alpha: str | Path | np.ndarray,
        cur_alpha: str | Path | np.ndarray,
        outputs: Mapping[str, Path],
        result: Any,
    ) -> None:
        """Overwrite the two output JSONs with CPU frame-diff metrics."""
        from ...eval.temporal_metrics import (
            frame_diff_flow,
            temporal_flicker,
            temporal_warped_error,
        )

        prev_arr = self._load_alpha(prev_alpha)
        cur_arr = self._load_alpha(cur_alpha)
        seq = [prev_arr, cur_arr]
        flicker = temporal_flicker(seq)["mean_flicker"]
        warped = temporal_warped_error(seq, flow_fn=frame_diff_flow)["mean_warped_error"]
        # flow_consistency_score: 1 - flicker (1 = perfectly stable, 0 = unstable).
        consistency = max(0.0, 1.0 - flicker)
        outputs["temporal_error"].write_text(
            json.dumps({"temporal_error": float(warped), "flicker": float(flicker)}, indent=2),
            encoding="utf-8",
        )
        outputs["flow_consistency_score"].write_text(
            json.dumps({"flow_consistency_score": float(consistency)}, indent=2),
            encoding="utf-8",
        )
        # Mark the fallback in the result so callers can tell the metrics
        # came from the CPU path, not RAFT.
        result.missing_outputs = []
        result.returncode = 0
        result.stderr = (result.stderr or "") + "\n[cpu fallback: frame_diff_flow]"

    @staticmethod
    def _load_alpha(alpha: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(alpha, np.ndarray):
            return np.clip(alpha.astype(np.float32), 0.0, 1.0)
        from PIL import Image

        return np.asarray(Image.open(str(alpha)).convert("L"), dtype=np.float32) / 255.0