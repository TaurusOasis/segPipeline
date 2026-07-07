"""XMem VOS adapter — long-video memory-based masklet fallback (step 4).

Wraps the :mod:`hmp.adapters.templates` catalog entry ``xmem``. The default
command template invokes ``python -m xmem.infer`` from a standard
hkchengrex/XMem checkout in a GPU env.

Outputs: ``masklet`` (per-frame mask directory) + ``track_id`` (JSON). XMem is
the long-video memory-based fallback used when Cutie's memory is insufficient
for very long sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["XMemAdapter"]

INTEGRATION = "xmem"


class XMemAdapter(SubprocessAdapter):
    """Typed adapter for external XMem long-video VOS masklet propagation."""

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
            "masklet": out / "masklet",
            "track_id": out / "track_id.json",
        }

    def propagate(
        self,
        image_dir: str | Path,
        mask_prompt: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Propagate a mask prompt across a long frame sequence; return (result, paths)."""
        outputs = self._output_paths(output_dir)
        inputs = {"image_dir": str(image_dir), "mask_prompt": str(mask_prompt)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs