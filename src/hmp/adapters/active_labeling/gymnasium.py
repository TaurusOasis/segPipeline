"""Gymnasium external adapter — RL env rollout for active labeling.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``gymnasium``. The
default command template invokes ``python -m hmp.adapters.active_labeling.gym_env``
which runs a Gymnasium custom env (state = unlabeled-instance pool +
model uncertainty, action = which instance to label next) and writes the
rollout episode + reward trace.

Outputs: ``agent_episode`` (the (state, action, reward) trajectory JSON),
``reward_trace`` (per-step reward JSON for tuning the reward shape).
Gymnasium is the priority-1 active-labeling adapter (env definition).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["GymnasiumAdapter"]

INTEGRATION = "gymnasium"

OUTPUT_FILES = {
    "agent_episode": "agent_episode.json",
    "reward_trace": "reward_trace.json",
}


class GymnasiumAdapter(SubprocessAdapter):
    """Typed adapter for the Gymnasium RL-env rollout (active labeling)."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 1800.0,
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

    def rollout(
        self,
        env_config: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Run a Gymnasium env rollout from env_config; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"env_config": str(env_config)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs