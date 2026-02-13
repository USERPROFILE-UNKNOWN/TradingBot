"""AI Master Agent orchestration with guarded paper/live controls."""

from __future__ import annotations

import os
from .event_bus import EventBus
from .scheduler import JobScheduler
from .governance import Governance
from .metrics import MetricsStore


class AgentMaster:
    MODES = ("OFF", "ADVISORY", "PAPER", "LIVE")

    def __init__(self, config, db_manager, log_callback=None):
        self.config = config
        self.db = db_manager
        self.log = log_callback or (lambda *_args, **_kwargs: None)
        self.mode = self._read_mode()
        self.bus = EventBus()
        self.gov = Governance(config)
        self.scheduler = JobScheduler(log_callback=self.log)

        db_dir = getattr(db_manager, "db_dir", None) or os.path.join(os.getcwd(), "db")
        self.metrics = MetricsStore(db_dir)

        self.bus.subscribe("*", self._on_event)
        self._register_jobs()
        self.scheduler.start()

    def shutdown(self):
        try:
            self.scheduler.stop()
        except Exception:
            pass

    def set_mode(self, mode: str):
        candidate = str(mode or "OFF").strip().upper()
        if candidate not in self.MODES:
            return False
        self.mode = candidate
        try:
            self.config["CONFIGURATION"]["agent_mode"] = candidate
        except Exception:
            pass
        self.log(f"🧠 [AGENT] Mode set to {candidate}")
        return True

    def publish(self, event_type: str, payload: dict | None = None):
        self.bus.publish(event_type, payload or {})

    def evaluate_action(self, action: dict):
        action_type = str(action.get("type", "UNKNOWN"))
        scope = "LIVE" if self.mode == "LIVE" else "PAPER"
        action["scope"] = scope

        if self.mode == "OFF":
            self.metrics.log_agent_action(self.mode, action_type, False, "Agent mode OFF", action)
            return False, "Agent mode OFF"

        if self.mode == "ADVISORY":
            self.metrics.log_agent_action(self.mode, action_type, False, "Advisory-only; no execution", action)
            return False, "Advisory-only"

        approved, reason = self.gov.approve_action(action)
        self.metrics.log_agent_action(self.mode, action_type, approved, reason, action)
        return approved, reason

    def _register_jobs(self):
        self.scheduler.add_job("agent_health_snapshot", 60, self._health_snapshot)

    def _health_snapshot(self):
        score = 1.0
        if self.mode == "OFF":
            score = 0.5
        elif self.mode == "ADVISORY":
            score = 0.7
        elif self.mode == "PAPER":
            score = 0.85
        self.metrics.log_metric("system_health_score", score, {"mode": self.mode})

    def _read_mode(self):
        try:
            m = str(self.config.get("CONFIGURATION", "agent_mode", fallback="OFF")).strip().upper()
        except Exception:
            m = "OFF"
        return m if m in self.MODES else "OFF"

    def _on_event(self, event: dict):
        ev_type = str(event.get("type", ""))
        payload = event.get("payload", {}) or {}

        if ev_type in {"ORDER_REJECTED", "RISK_BREACH", "DATA_GAP"}:
            self.metrics.log_metric("anomaly_count", 1.0, {"event_type": ev_type, **payload})
            if self.mode in {"PAPER", "LIVE"}:
                self.log(f"🧠 [AGENT] Detected {ev_type}; tightening guardrails.")

        if ev_type == "ACTION_REQUEST":
            approved, reason = self.evaluate_action(payload)
            self.log(f"🧠 [AGENT] Action {payload.get('type', 'UNKNOWN')}: {'APPROVED' if approved else 'DENIED'} ({reason})")
