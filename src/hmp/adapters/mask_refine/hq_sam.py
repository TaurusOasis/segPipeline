"""SAM-HQ external adapter — box-prompted boundary refinement.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``hq_sam`` with a typed
:meth:`HqSamAdapter.refine` helper. The default command template invokes
``python -m sam_hq.predict`` from a standard SysCV/sam-hq checkout; override
``command_template`` to point at a different entrypoint or a mock command for
CPU tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["HqSamAdapter"]

INTEGRATION = "hq_sam"


class HqSamAdapter(SubprocessAdapter):
    """Typed adapter for the external SAM-HQ repo (box prompt)."""

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

    def refine(
        self,
        image: str | Path,
        box: str,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Refine a box prompt on ``image``; return (result, output_paths).

        ``box`` is a free-form string the external CLI understands (typically
        ``"x1,y1,x2,y2"``). Output is ``refined_mask.png`` under ``output_dir``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        outputs = {"refined_mask": out / "refined_mask.png"}
        inputs = {"image": str(image), "box": str(box)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs