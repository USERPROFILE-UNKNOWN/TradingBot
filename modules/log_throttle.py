"""Throttled logging helpers.

Goal (v5.12.3 updateA): reduce "silent failure" by logging exceptions with enough
context to diagnose, while preventing spam loops from flooding the runtime log.

Design notes:
- The project uses multiple logging styles (self.log(str), print, engine._emit).
  These helpers therefore accept any callable that takes a single string.
- Uses monotonic time so system clock changes don't break throttling.
"""

from __future__ import annotations

import threading
import time
import os
import traceback
from typing import Any, Callable, Dict, Optional


_LogFn = Callable[[str], Any]


_LAST: Dict[str, float] = {}
_LOCK = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _safe_str(x: Any, max_len: int = 300) -> str:
    try:
        s = str(x)
    except Exception:
        try:
            s = repr(x)
        except Exception:
            s = "<unprintable>"
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _emit(log_fn: Optional[_LogFn], msg: str) -> None:
    if not log_fn:
        return
    try:
        log_fn(msg)
    except Exception:
        # Never allow logging failures to crash the caller.
        pass


def should_emit(key: str, throttle_sec: float) -> bool:
    """Return True if enough time has elapsed since the last emit for key."""
    if not key:
        return True
    try:
        throttle = float(throttle_sec)
    except Exception:
        throttle = 0.0

    if throttle <= 0:
        return True

    t = _now()
    with _LOCK:
        last = _LAST.get(key)
        if last is None or (t - last) >= throttle:
            _LAST[key] = t
            return True
    return False


def log_throttled(
    log_fn: Optional[_LogFn],
    event_id: str,
    message: str,
    *,
    key: Optional[str] = None,
    throttle_sec: float = 300.0,
) -> None:
    """Emit a throttled log line.

    Args:
        log_fn: A callable that accepts a string (e.g., TradingApp.log).
        event_id: Stable event identifier (E_*).
        message: Human-readable message.
        key: Throttle key (defaults to event_id).
        throttle_sec: Minimum seconds between emits per key.
    """
    ev = _safe_str(event_id or "E_UNKNOWN", 60)
    k = key or ev
    if not should_emit(k, throttle_sec):
        return
    _emit(log_fn, f"[{ev}] {message}")


def log_exception_throttled(
    log_fn: Optional[_LogFn],
    event_id: str,
    exc: BaseException,
    *,
    key: Optional[str] = None,
    throttle_sec: float = 300.0,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a throttled exception summary with optional context."""
    ctx = ""
    if context:
        parts = []
        for k, v in context.items():
            parts.append(f"{k}={_safe_str(v, 120)}")
        if parts:
            ctx = " | " + " ".join(parts)

    where = ""
    try:
        tb = getattr(exc, "__traceback__", None)
        if tb:
            frames = traceback.extract_tb(tb)
            if frames:
                fr = frames[-1]
                fn = os.path.basename(fr.filename or "")
                where = f" | at={fn}:{fr.lineno} {fr.name}()"
    except Exception:
        where = ""

    et = type(exc).__name__ if exc is not None else "Exception"
    em = _safe_str(exc, 400)
    msg = f"{et}: {em}{where}{ctx}"
    log_throttled(log_fn, event_id, msg, key=key or event_id, throttle_sec=throttle_sec)
