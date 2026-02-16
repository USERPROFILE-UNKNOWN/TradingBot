"""Engine entrypoint (coordinator-only).

v5.15.4: core.py is intentionally thin glue.
The full TradingEngine implementation lives in core_impl.py (forward-only refactor).

This split keeps imports stable (modules.engine.core.TradingEngine) while allowing
the engine core file to remain a small orchestration surface.
"""

from __future__ import annotations

import importlib
import time  # NOTE: tests monkeypatch core.time.time

from .broker_gateway import BrokerGateway  # tests monkeypatch BrokerGateway.connect

# Reload implementation on core reload to keep tests deterministic when they stub deps.
from . import core_impl as _core_impl

try:
    _core_impl = importlib.reload(_core_impl)
except Exception:
    # Fail-safe: if reload fails, keep the imported module to avoid import-time crashes.
    pass

TradingEngine = _core_impl.TradingEngine

__all__ = [
    "TradingEngine",
    "BrokerGateway",
    "time",
]
