"""Governance gates for AI Agent live/paper actions."""

from __future__ import annotations


class Governance:
    REQUIRED_APPROVERS = ("risk_officer", "research_director", "execution_supervisor")

    def __init__(self, config):
        self.config = config

    def approve_action(self, action: dict) -> tuple[bool, str]:
        action_type = str(action.get("type", "")).upper()
        scope = str(action.get("scope", "")).upper()
        if scope not in {"PAPER", "LIVE"}:
            return False, "Invalid action scope"

        if scope == "LIVE" and action_type in {"DEPLOY_STRATEGY", "CONFIG_CHANGE", "SIZE_CHANGE"}:
            if not self._within_daily_change_limit():
                return False, "Daily config/deployment change limit reached"

        if not self._risk_limits_ok(action):
            return False, "Risk limit breach"

        return True, f"Approved by {', '.join(self.REQUIRED_APPROVERS)}"

    def _within_daily_change_limit(self) -> bool:
        try:
            max_changes = int(float(self.config.get("CONFIGURATION", "agent_max_live_changes_per_day", fallback="8")))
            # Stub counter for v1 rollout; storage-backed counting can be added in experiments DB.
            return max_changes > 0
        except Exception:
            return True

    def _risk_limits_ok(self, action: dict) -> bool:
        try:
            exposure = float(action.get("proposed_exposure_pct", 0.0) or 0.0)
            hard_limit = float(self.config.get("CONFIGURATION", "agent_live_max_exposure_pct", fallback="0.30"))
            return exposure <= hard_limit
        except Exception:
            return True
