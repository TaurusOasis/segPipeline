"""Stable-Baselines3 external adapter — RL agent training/rollback for active labeling.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``stable_baselines3``.
The default command template invokes ``python -m hmp.adapters.active_labeling.sb3_agent``
which trains (or loads) a Stable-Baselines3 policy on the Gymnasium env and
writes the checkpoint + decision trace.

Outputs: ``policy_checkpoint`` (the trained policy weights, .zip),
``decision_trace`` (per-instance label/no-label decisions JSON).
Stable-Baselines3 is the priority-2 active-labeling adapter (the agent
that consumes the Gymnasium env).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["StableBaselines3Adapter"]

INTEGRATION = "stable_baselines3"

OUTPUT_FILES = {
    "policy_checkpoint": "policy_checkpoint.zip",
    "decision_trace": "decision_trace.json",
}


class StableBaselines3Adapter(SubprocessAdapter):
    """Typed adapter for the Stable-Baselines3 RL agent (active labeling)."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 3600.0,
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

    def train(
        self,
        env_config: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Train/load an SB3 policy from env_config; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"env_config": str(env_config)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs