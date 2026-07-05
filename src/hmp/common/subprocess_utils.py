"""Safe subprocess helpers for external research-repo adapters."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .logging import get_logger

log = get_logger("hmp.common.subprocess")


def render_command(template: str, values: Mapping[str, object]) -> str:
    """Format a command template with ``str.format``-style placeholders."""
    return str(template).format(**{k: str(v) for k, v in values.items()})


def run_command(
    command: str,
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command string and return the completed process."""
    log.info("command: %s", command)
    if dry_run:
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {command}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc


def split_argv(command: str) -> list[str]:
    return shlex.split(command)


def validate_outputs(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing expected outputs: {missing}")
