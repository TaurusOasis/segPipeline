"""JSONL read/write helpers (Step 01).

Works with plain dicts or pydantic models. Deterministic line ordering:
records are written in the order given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Record = Union[BaseModel, dict]


def _to_dict(record: Record) -> dict:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def write_jsonl(
    path: str | Path | Iterable[Record],
    records: Iterable[Record] | str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Write records to ``path`` as JSON lines.

    If ``overwrite`` is False and ``path`` exists, raise ``FileExistsError``.
    Parent directories are created automatically.
    """
    if not isinstance(path, (str, Path)) and isinstance(records, (str, Path)):
        path, records = records, path
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use overwrite=True): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_to_dict(record), ensure_ascii=False))
            fh.write("\n")


def read_jsonl(
    path: str | Path,
    model: Optional[Type[T]] = None,
) -> Iterator[Union[dict, T]]:
    """Yield records from a JSONL file.

    If ``model`` is given, each line is parsed/validated into that pydantic
    model; otherwise raw dicts are yielded. Blank lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if model is not None:
                yield model.model_validate(obj)
            else:
                yield obj


def read_jsonl_list(path: str | Path, model: Optional[Type[T]] = None) -> list:
    """Eager version of :func:`read_jsonl` returning a list."""
    return list(read_jsonl(path, model=model))


def count_jsonl(path: str | Path) -> int:
    """Count non-blank lines in a JSONL file without parsing."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n
