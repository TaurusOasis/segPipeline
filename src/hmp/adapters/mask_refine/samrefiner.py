"""SAMRefiner external adapter — coarse-mask boundary refinement.

Wraps the :mod:`hmp.adapters.templates` catalog entry ``samrefiner`` with a
typed :meth:`SamRefinerAdapter.refine` helper and a batch runner
:meth:`SamRefinerAdapter.refine_batch` that writes a provenance JSONL.

The default command template invokes ``python -m samrefiner.refine`` from a
standard SAMRefiner checkout; override ``command_template`` at construction
to point at a different entrypoint or to a mock command for CPU tests. The
``repo_python`` parameter selects the interpreter for the external repo's env
(e.g. a ``yolo26-cu133`` venv).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..base import AdapterRegistry, SubprocessAdapter, load_registry
from ..templates import template_for

__all__ = ["SamRefinerAdapter"]

INTEGRATION = "samrefiner"


class SamRefinerAdapter(SubprocessAdapter):
    """Typed adapter for the external SAMRefiner repo."""

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
            "refined_mask": out / "refined_mask.png",
            "mask_quality": out / "mask_quality.json",
        }

    def refine(
        self,
        image: str | Path,
        coarse_mask: str | Path,
        *,
        output_dir: str | Path,
        execute: bool = True,
    ) -> "tuple[Any, dict[str, Path]]":
        """Refine ``coarse_mask`` against ``image``; return (result, output_paths).

        ``execute=True`` runs the subprocess (real GPU env or mock command);
        ``execute=False`` only resolves the dry run. ``output_dir`` is
        created and the two output paths (``refined_mask.png``,
        ``mask_quality.json``) are placed inside it.
        """
        outputs = self._output_paths(output_dir)
        inputs = {"image": str(image), "coarse_mask": str(coarse_mask)}
        params = {"repo_python": self.repo_python}
        if execute:
            result = self.run(inputs, outputs, params=params)
        else:
            result = self.dry_run(inputs, outputs, params=params)
        return result, outputs

    def refine_batch(
        self,
        items: Iterable[Mapping[str, str]],
        *,
        output_root: str | Path,
        execute: bool = True,
        provenance_path: Optional[str | Path] = None,
    ) -> list[dict[str, object]]:
        """Refine many (image, coarse_mask) pairs; optionally write provenance.

        Each item must carry ``image``, ``coarse_mask`` and (recommended)
        ``item_id`` + ``instance_id`` for provenance. Per-item outputs go
        under ``output_root/<item_id or index>/``. Returns a list of
        provenance rows (one per item), and writes them as JSONL to
        ``provenance_path`` when given.
        """
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for i, item in enumerate(items):
            item_id = str(item.get("item_id") or f"item{i}")
            instance_id = str(item.get("instance_id") or f"inst{i}")
            out_dir = root / item_id
            result, outputs = self.refine(
                item["image"],
                item["coarse_mask"],
                output_dir=out_dir,
                execute=execute,
            )
            prov = self.provenance(
                result,
                branch_source={
                    "stage": "mask_refine",
                    "item_id": item_id,
                    "instance_id": instance_id,
                },
            )
            row: dict[str, object] = {
                "item_id": item_id,
                "instance_id": instance_id,
                "ok": result.ok,
                "returncode": result.returncode,
                "dry_run": result.dry_run,
                "missing_outputs": result.missing_outputs,
                "outputs": {k: str(v) for k, v in outputs.items()},
                **prov,
            }
            rows.append(row)

        if provenance_path is not None:
            pp = Path(provenance_path)
            pp.parent.mkdir(parents=True, exist_ok=True)
            with open(pp, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows