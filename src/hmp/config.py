"""Configuration loading for hmp.

Configs are plain YAML files. They are loaded into a :class:`Config` object that
behaves like a nested dict with attribute access. No heavy dependencies are
imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml


class Config:
    """Attribute-access wrapper around a nested dict.

    ``cfg.paths.raw_dir`` works for ``{"paths": {"raw_dir": "data/raw"}}``.
    Missing keys raise :class:`AttributeError` with a helpful message.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", data or {})

    # -- mapping helpers -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(
                f"Config has no key {name!r}; available: {sorted(self._data)}"
            ) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, name: str) -> Any:
        value = self._data[name]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file from ``path`` into a :class:`Config`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return Config(data)


def resolve_path(base: str | Path, rel: str | Path) -> Path:
    """Resolve ``rel`` against ``base`` directory.

    If ``rel`` is absolute it is returned unchanged. Otherwise it is joined
    to ``base``. ``base`` may itself be a relative path (resolved against CWD).
    """
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    return (Path(base) / rel).resolve() if Path(base).is_absolute() else Path(base) / rel


def seed_from_config(cfg: Config) -> int:
    """Return the project seed (default 42)."""
    return int(cfg.get("project", {}).get("seed", 42) or 42)