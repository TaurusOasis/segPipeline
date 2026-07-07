"""Label Studio external adapter — human-edit bridge.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``label_studio``. The
default command template invokes ``python -m hmp.adapters.hitl.label_studio_bridge``
which exports the human edits for a Label Studio project and writes the
audit log.

Outputs (each a JSON file): ``human_edits`` (the corrected masks/boxes),
``audit_log`` (who/when/what trail). Label Studio is the alternative HITL
correction adapter (non-CVAT shops).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["LabelStudioAdapter"]

INTEGRATION = "label_studio"

OUTPUT_FILES = {
    "human_edits": "human_edits.json",
    "audit_log": "audit_log.json",
}


class LabelStudioAdapter(SubprocessAdapter):
    """Typed adapter for the Label Studio human-edit bridge."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        project_id: str = "",
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
        self.project_id = project_id

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def correct(
        self,
        *,
        output_dir: str | Path,
        project_id: Optional[str] = None,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Export human edits for a Label Studio project; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs: dict[str, str] = {}
        params = {"repo_python": self.repo_python, "project_id": project_id or self.project_id}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs