"""Logging setup for hmp.

A small wrapper around the stdlib ``logging`` module so every stage gets a
consistently formatted logger without importing anything heavy.
"""

from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED: bool = False


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once.

    Safe to call multiple times; only the first call installs the handler.
    """
    global _CONFIGURED
    if isinstance(level, str):
        level = level.upper()
    numeric = logging.getLevelName(level) if isinstance(level, str) else int(level)
    if isinstance(numeric, str):  # unknown level name -> fallback
        numeric = logging.INFO

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        _CONFIGURED = True
    logging.getLogger().setLevel(numeric)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger; auto-configures at INFO if nothing is configured."""
    if not _CONFIGURED:
        configure_logging("INFO")
    return logging.getLogger(name if name else "hmp")