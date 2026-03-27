"""Shared logging utilities for application scripts and services."""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_LOG_LEVEL = "INFO"


def setup_logging(level: str | None = None) -> None:
    """Configure process-wide logging with a consistent format.

    The function is safe to call from multiple entry points because
    ``logging.basicConfig(..., force=True)`` replaces any previous default
    configuration in the current process.
    """

    resolved_level = (level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that follows the shared project configuration."""

    return logging.getLogger(name)
