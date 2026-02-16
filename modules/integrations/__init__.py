"""Integration modules (forward-only).

Currently:
- TradingView webhook ingestion (Phase 1.5)

This package is intentionally lightweight and stdlib-only to keep PyInstaller
bundling stable.
"""

from __future__ import annotations

__all__ = [
    "tradingview_webhook",
]
