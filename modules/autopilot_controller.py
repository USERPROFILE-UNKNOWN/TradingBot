"""Autopilot lifecycle controller.

v6.21.2: centralizes explicit start/stop and shared state-write lock wiring.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class AutopilotController:
    """Controls AgentMaster automation lifecycle and mutation safeguards."""

    def __init__(self, agent_master, *, log_fn: Optional[Callable[[str], None]] = None):
        self.agent_master = agent_master
        self.log = log_fn or (lambda *_a, **_k: None)
        self._state_write_lock = threading.RLock()
        self._running = False

    def start(self, *, engine_running_provider: Optional[Callable[[], bool]] = None) -> bool:
        if self._running:
            return False
        try:
            if hasattr(self.agent_master, "attach_state_controls"):
                self.agent_master.attach_state_controls(
                    state_write_lock=self._state_write_lock,
                    engine_running_provider=engine_running_provider,
                )
            started = bool(self.agent_master.start())
            self._running = started
            return started
        except Exception as e:
            self.log(f"[AGENT] Autopilot controller failed to start: {e}")
            return False

    def stop(self) -> None:
        try:
            self.agent_master.shutdown()
        finally:
            self._running = False

    @property
    def running(self) -> bool:
        return bool(self._running)
