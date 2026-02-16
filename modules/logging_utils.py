"""Centralized logging configuration helpers.

Keeps logging setup consistent across startup/runtime modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


class _ContextDefaultsFilter(logging.Filter):
    """Ensure commonly-used structured fields exist on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "component"):
            record.component = "app"
        if not hasattr(record, "symbol"):
            record.symbol = "-"
        if not hasattr(record, "order_id"):
            record.order_id = "-"
        if not hasattr(record, "strategy"):
            record.strategy = "-"
        if not hasattr(record, "mode"):
            record.mode = "-"
        return True


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

    context_filter = _ContextDefaultsFilter()
    for handler in handlers:
        handler.addFilter(context_filter)

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "component=%(component)s mode=%(mode)s symbol=%(symbol)s order_id=%(order_id)s strategy=%(strategy)s | "
            "%(message)s"
        ),
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_component_logger(name: str, component: str) -> logging.LoggerAdapter:
    """Return a logger adapter that injects the component field."""
    return logging.LoggerAdapter(get_logger(name), {"component": component})
