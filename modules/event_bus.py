"""Lightweight in-process event bus for TradingBot agent orchestration."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable, Dict, List, Any


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[dict], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable[[dict], None]) -> None:
        if not event_type or not callable(callback):
            return
        with self._lock:
            self._subs[event_type].append(callback)

    def publish(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        if not event_type:
            return
        event = {"type": event_type, "payload": payload or {}}
        with self._lock:
            listeners = list(self._subs.get(event_type, [])) + list(self._subs.get("*", []))
        for cb in listeners:
            try:
                cb(event)
            except Exception:
                continue
