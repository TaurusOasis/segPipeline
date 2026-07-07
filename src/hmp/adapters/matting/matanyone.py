"""MatAnyone external adapter — target-assigned video matting (branch Bv).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``matanyone``. The
default command template invokes ``python -m matanyone.infer`` from a
standard pq-yang/MatAnyone checkout in a GPU env.

Outputs: ``alpha_video`` (the matte, a video file or directory depending on
the repo) and ``branch_source`` (a small JSON recording the branch provenance
so :meth:`validate_outputs` passes against the registry ``expected_outputs``
``["alpha_video", "branch_source"]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["MatAnyoneAdapter"]

INTEGRATION = "matanyone"


class MatAnyoneAdapter(SubprocessAdapter):
    """Typed adapter for external MatAnyone video matting (branch Bv)."""

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
            "branch_source": out / "branch_source.json",
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
        """Run target-assigned video matting; return (result, output_paths).

        When ``execute=True`` and the external repo succeeds, the
        ``branch_source.json`` is written by the external CLI. When the CLI
        fails (e.g. repo not present in CPU env), :meth:`mat` writes a minimal
        ``branch_source.json`` itself so the contract is still inspectable —
        the ``alpha_video`` output is left missing (``missing_outputs`` will
        list it) so the caller can route to review.
        """
        outputs = self._output_paths(output_dir)
        inputs = {"image_dir": str(image_dir), "target_mask": str(target_mask)}
        params = {"repo_python": self.repo_python}
        if not execute:
            result = self.dry_run(inputs, outputs, params=params)
            return result, outputs

        result = self.run(inputs, outputs, params=params)
        # Always ensure branch_source.json exists (the external CLI may or may
        # not write it). If it's missing, write a minimal provenance stub so
        # downstream readers can identify the branch even on failure.
        bs = outputs["branch_source"]
        if not bs.exists():
            bs.write_text(
                json.dumps(
                    {
                        "adapter": self.spec.name,
                        "branch": "Bv",
                        "target_id": target_id,
                        "image_dir": str(image_dir),
                        "returncode": result.returncode,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            # Re-validate now that branch_source exists.
            result.missing_outputs = self.validate_outputs(outputs)
        return result, outputs