"""Centralized logging configuration helpers.

Keeps logging setup consistent across startup/runtime modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logging once with console + optional file sink."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(os.path.join(log_dir, "tradingbot.log"), encoding="utf-8"))
        except Exception:
            # Best effort: fallback to console-only logging.
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
