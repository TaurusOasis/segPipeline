"""Base contract for external research-repo subprocess adapters.

Design rules (from ``doc/OPEN_SOURCE_INTEGRATION_TARGETS_zh.md`` /
``configs/reference_integrations.yaml``):

* ``integration_style: adapter_or_subprocess`` — external repos are invoked
  as subprocesses via a command template, never vendored.
* ``require_dry_run: true`` — every adapter must be able to produce its
  resolved command + env + input/output map *without* executing, so the
  pipeline can preview, log, and license-check before running.
* Outputs flow back into hmp's JSONL schema; provenance
  (``prompt_history`` / ``branch_source`` / ``quality_scores`` /
  ``license_meta``) is emitted by :meth:`ExternalAdapter.provenance`.

Command templates are argv lists (not shell strings) for safety. Placeholders
use a flat namespace: ``{input_<key>}`` for inputs and ``{output_<key>}`` for
outputs, plus any ``{param_<key>}``. For example, a SAMRefiner adapter might
use::

    ["python", "-m", "samrefiner.cli",
     "--image", "{input_image}",
     "--coarse-mask", "{input_coarse_mask}",
     "--output", "{output_refined_mask}"]
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ..common.logging import get_logger

log = get_logger("hmp.adapters")

# Default registry path, relative to repo root.
DEFAULT_REGISTRY_PATH = "configs/reference_integrations.yaml"


class MissingPlaceholder(KeyError):
    """Raised when a command template references an unset placeholder."""


class _StrictFormatDict(dict):
    """Mapping that errors loudly on missing template placeholders."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - via format_map
        raise MissingPlaceholder(key)


@dataclass
class AdapterSpec:
    """Declarative metadata for one external integration.

    Mirrors an entry under ``integrations:`` in
    ``configs/reference_integrations.yaml``.
    """

    name: str
    group: str
    url: str
    adapter_target: str
    expected_outputs: list[str] = field(default_factory=list)
    role: str = ""
    priority: int = 5
    license_review: str = "required"

    @classmethod
    def from_registry_entry(cls, name: str, entry: Mapping[str, Any]) -> "AdapterSpec":
        return cls(
            name=name,
            group=str(entry.get("group", "")),
            url=str(entry.get("url", "")),
            adapter_target=str(entry.get("adapter_target", "")),
            expected_outputs=list(entry.get("expected_outputs", []) or []),
            role=str(entry.get("role", "")),
            priority=int(entry.get("priority", 5)),
            license_review=str(entry.get("license_review", "required")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterResult:
    """Outcome of an adapter run (real or dry).

    ``outputs`` maps the spec's ``expected_outputs`` keys (or the caller's
    output dict keys) to concrete filesystem paths.
    """

    name: str
    command: list[str]
    env: dict[str, str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    outputs: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    missing_outputs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.missing_outputs


class ExternalAdapter:
    """Base class for all external-repo adapters.

    Subclasses provide a command template (and optionally override
    :meth:`build_command` / :meth:`validate_outputs` / :meth:`provenance`).
    The base handles dry-run, subprocess execution, timeout, output
    validation, and provenance emission.
    """

    def __init__(
        self,
        spec: AdapterSpec,
        *,
        workdir: str | Path,
        command_template: list[str] | None = None,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.command_template = list(command_template or [])
        self.env_overlay: dict[str, str] = dict(env or {})
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------------ #
    # Command building
    # ------------------------------------------------------------------ #
    def _format_namespace(
        self,
        inputs: Mapping[str, str | Path],
        outputs: Mapping[str, str | Path],
        params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, str]:
        ns: dict[str, str] = {}
        for k, v in inputs.items():
            ns[f"input_{k}"] = str(v)
        for k, v in outputs.items():
            ns[f"output_{k}"] = str(v)
        for k, v in (params or {}).items():
            ns[f"param_{k}"] = str(v)
        return ns

    def build_command(
        self,
        inputs: Mapping[str, str | Path],
        outputs: Mapping[str, str | Path],
        params: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        """Resolve the command template against inputs/outputs/params.

        Unknown placeholders raise :class:`MissingPlaceholder`. Literal
        braces in a token should be escaped as ``{{`` / ``}}``.
        """
        if not self.command_template:
            raise ValueError(
                f"adapter {self.spec.name!r} has no command_template set"
            )
        ns = _StrictFormatDict(self._format_namespace(inputs, outputs, params))
        resolved: list[str] = []
        for token in self.command_template:
            try:
                resolved.append(token.format_map(ns))
            except MissingPlaceholder as exc:
                raise MissingPlaceholder(
                    f"adapter {self.spec.name!r}: placeholder {{{exc.args[0]}}} "
                    f"referenced by token {token!r} was not supplied"
                ) from None
        return resolved

    # ------------------------------------------------------------------ #
    # Dry run
    # ------------------------------------------------------------------ #
    def dry_run(
        self,
        inputs: Mapping[str, str | Path],
        outputs: Mapping[str, str | Path],
        params: Optional[Mapping[str, Any]] = None,
    ) -> AdapterResult:
        """Resolve command + env + outputs without executing anything."""
        cmd = self.build_command(inputs, outputs, params)
        env = self._resolved_env()
        log.info("[%s] dry-run: %s", self.spec.name, cmd)
        return AdapterResult(
            name=self.spec.name,
            command=cmd,
            env=env,
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
            outputs={k: str(v) for k, v in outputs.items()},
            dry_run=True,
            missing_outputs=[],
        )

    # ------------------------------------------------------------------ #
    # Real run
    # ------------------------------------------------------------------ #
    def _resolved_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env_overlay)
        return env

    def run(
        self,
        inputs: Mapping[str, str | Path],
        outputs: Mapping[str, str | Path],
        params: Optional[Mapping[str, Any]] = None,
    ) -> AdapterResult:
        """Execute the adapter and validate outputs.

        On non-zero returncode the result is returned with ``ok=False`` rather
        than raising, so the pipeline can route the failure into QA/review.
        Output validation populates ``missing_outputs``.
        """
        cmd = self.build_command(inputs, outputs, params)
        env = self._resolved_env()
        log.info("[%s] run: %s", self.spec.name, cmd)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return AdapterResult(
                name=self.spec.name,
                command=cmd,
                env=env,
                returncode=-1,
                stdout=str(exc.stdout or ""),
                stderr=f"timeout after {self.timeout_s}s: {exc}",
                duration_s=duration,
                outputs={k: str(v) for k, v in outputs.items()},
                dry_run=False,
                missing_outputs=list(outputs.keys()),
            )
        duration = time.perf_counter() - start
        missing = self.validate_outputs(outputs)
        return AdapterResult(
            name=self.spec.name,
            command=cmd,
            env=env,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            outputs={k: str(v) for k, v in outputs.items()},
            dry_run=False,
            missing_outputs=missing,
        )

    # ------------------------------------------------------------------ #
    # Output validation
    # ------------------------------------------------------------------ #
    def validate_outputs(self, outputs: Mapping[str, str | Path]) -> list[str]:
        """Return the list of expected output keys whose files do not exist.

        Does not raise — the caller (or :meth:`run`) decides how to route the
        failure. ``expected_outputs`` from the spec, if any, are checked; if
        the spec lists none, all provided ``outputs`` are checked.
        """
        keys_to_check = self.spec.expected_outputs or list(outputs.keys())
        missing: list[str] = []
        for key in keys_to_check:
            path = outputs.get(key)
            if path is None:
                missing.append(key)
                continue
            if not Path(path).exists():
                missing.append(key)
        return missing

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #
    def provenance(
        self,
        result: AdapterResult,
        *,
        branch_source: Optional[Mapping[str, str]] = None,
        quality_scores: Optional[Mapping[str, float]] = None,
        extra_license_meta: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object]:
        """Emit the provenance fields the schema/QA layer consumes.

        Aligns with ``QualityRecord.branch_source`` / ``prompt_history`` /
        ``quality_scores`` / ``license_meta`` from
        :mod:`hmp.schemas` and the registry's ``provenance_fields`` policy.
        """
        branch = {"adapter": self.spec.name, "group": self.spec.group}
        if branch_source:
            branch.update(dict(branch_source))
        license_meta: dict[str, object] = {
            "license_review": self.spec.license_review,
            "url": self.spec.url,
        }
        if extra_license_meta:
            license_meta.update(dict(extra_license_meta))
        return {
            "branch_source": branch,
            "prompt_history": [
                {
                    "adapter": self.spec.name,
                    "command": result.command,
                    "dry_run": result.dry_run,
                    "returncode": result.returncode,
                    "outputs": result.outputs,
                }
            ],
            "quality_scores": dict(quality_scores or {}),
            "license_meta": license_meta,
        }


class SubprocessAdapter(ExternalAdapter):
    """Generic concrete adapter: a plain argv command template.

    This is the default adapter shape — used directly for integrations whose
    invocation is a normal command line. Specialized adapters (e.g. a future
    ``Sam2Adapter`` that lazy-imports ``ultralytics`` in-process) subclass
    :class:`ExternalAdapter` and override :meth:`run`.
    """


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
class AdapterRegistry:
    """Lookup table of :class:`AdapterSpec` loaded from the yaml registry."""

    def __init__(self, specs: Mapping[str, AdapterSpec]) -> None:
        self.specs: dict[str, AdapterSpec] = dict(specs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AdapterRegistry":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        entries = data.get("integrations", {}) or {}
        specs = {
            name: AdapterSpec.from_registry_entry(name, entry)
            for name, entry in entries.items()
        }
        return cls(specs)

    def get(self, name: str) -> AdapterSpec:
        if name not in self.specs:
            raise KeyError(f"unknown adapter integration: {name!r}")
        return self.specs[name]

    def names(self) -> list[str]:
        return list(self.specs.keys())

    def by_group(self, group: str) -> list[AdapterSpec]:
        return [s for s in self.specs.values() if s.group == group]

    def build(
        self,
        name: str,
        *,
        workdir: str | Path,
        command_template: list[str],
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 600.0,
    ) -> SubprocessAdapter:
        """Construct a generic :class:`SubprocessAdapter` for ``name``."""
        return SubprocessAdapter(
            self.get(name),
            workdir=workdir,
            command_template=command_template,
            env=env,
            timeout_s=timeout_s,
        )


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> AdapterRegistry:
    """Load the adapter registry from ``configs/reference_integrations.yaml``."""
    return AdapterRegistry.from_yaml(path)