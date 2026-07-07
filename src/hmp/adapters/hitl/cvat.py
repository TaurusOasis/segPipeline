"""CVAT external adapter — human-edit / corrected-prompt bridge.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``cvat``. The default
command template invokes ``python -m hmp.adapters.hitl.cvat_bridge`` which
exports the human edits for a CVAT task and writes the corrected prompts +
audit log.

Outputs (each a JSON file): ``human_edits`` (the corrected masks/boxes),
``corrected_prompts`` (the reviewer-corrected point/box/scribble prompts,
fed back into the prompt agent), ``audit_log`` (who/when/what trail).
CVAT is the priority HITL correction adapter for mask/prompt fixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["CvatAdapter"]

INTEGRATION = "cvat"

OUTPUT_FILES = {
    "human_edits": "human_edits.json",
    "corrected_prompts": "corrected_prompts.json",
    "audit_log": "audit_log.json",
}


class CvatAdapter(SubprocessAdapter):
    """Typed adapter for the CVAT human-edit / corrected-prompt bridge."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        task_id: str = "",
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
        self.task_id = task_id

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def correct(
        self,
        *,
        output_dir: str | Path,
        task_id: Optional[str] = None,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Export human edits for a CVAT task; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs: dict[str, str] = {}
        params = {"repo_python": self.repo_python, "task_id": task_id or self.task_id}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs