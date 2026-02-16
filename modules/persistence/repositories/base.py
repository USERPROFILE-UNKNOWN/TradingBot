from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Any

try:
    from ..db_runtime import DbRuntime
except Exception:  # pragma: no cover
    DbRuntime = Any  # type: ignore


class RepoBase:
    """Thin repository base around DbRuntime.

    Repos are intentionally dumb: they wrap the current SQL behavior without
    changing semantics. DataManager remains the public façade until v5.16.2.
    """

    def __init__(
        self,
        runtime: Optional["DbRuntime"],
        *,
        split_mode: bool = True,
        conn_fallback=None,
        lock_fallback: Optional[threading.Lock] = None,
        logger=None,
        log_fn: Optional[Callable[[str], None]] = None,
        read_agent_mode: Optional[Callable[[], str]] = None,
    ):
        self.runtime = runtime
        self.split_mode = bool(split_mode)
        self._conn_fallback = conn_fallback
        # RLock prevents self-deadlocks when higher-level code composes repo calls.
        self._lock_fallback = lock_fallback or threading.RLock()
        self._logger = logger
        self._log_fn = log_fn
        self._read_agent_mode = read_agent_mode or (lambda: "OFF")

    def _log(self, msg: str, **context):
        try:
            if self._log_fn:
                # DataManager._log accepts (msg, **context)
                self._log_fn(msg, **context)  # type: ignore[arg-type]
                return
        except Exception:
            pass
        try:
            if self._logger:
                self._logger.info(msg, extra=context if context else None)
        except Exception:
            pass

    def _conn(self, key: str):
        if self.runtime and self.split_mode:
            return self.runtime.conn(key)
        return self._conn_fallback

    def _lock(self, key: str):
        if self.runtime and self.split_mode:
            return self.runtime.lock(key)
        return self._lock_fallback

    @contextmanager
    def _multi_lock(self, keys: Iterable[str]):
        if self.runtime and self.split_mode:
            with self.runtime.multi_lock(keys):
                yield
            return
        # Single-lock fallback
        with self._lock_fallback:
            yield
