"""FiftyOne external adapter — dataset view + reviewer selection export.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``fiftyone``. The
default command template invokes ``python -m hmp.adapters.hitl.fiftyone_view``
which opens (or exports) a FiftyOne dataset view over the model outputs and
writes the reviewer's selection back to disk.

Outputs (each a JSON file): ``dataset_view`` (the view spec / state used),
``review_selection`` (the reviewer's kept/dropped/flagged selection).
FiftyOne is the data-management HITL adapter for browsing and selecting
which model-produced instances to keep for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["FiftyoneAdapter"]

INTEGRATION = "fiftyone"

OUTPUT_FILES = {
    "dataset_view": "dataset_view.json",
    "review_selection": "review_selection.json",
}


class FiftyoneAdapter(SubprocessAdapter):
    """Typed adapter for the FiftyOne dataset-view / review-selection bridge."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        repo_python: str = "python",
        view_spec: str = "default",
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
        self.view_spec = view_spec

    def _output_paths(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {k: out / fn for k, fn in OUTPUT_FILES.items()}

    def review(
        self,
        dataset_dir: str | Path,
        *,
        output_dir: str | Path,
        view_spec: Optional[str] = None,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Open/export a FiftyOne view over dataset_dir; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"dataset_dir": str(dataset_dir)}
        params = {"repo_python": self.repo_python, "view_spec": view_spec or self.view_spec}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs