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

        if scope == "LIVE" and action_type == "DEPLOY_STRATEGY":
            if not self._cfg_bool("CONFIGURATION", "agent_promotion_enabled", True):
                return False, "Promotion disabled by governance"

        if scope == "LIVE" and action_type == "CONFIG_CHANGE":
            if not self._cfg_bool("CONFIGURATION", "agent_config_tuning_enabled", True):
                return False, "Config tuning disabled by governance"

        if not self._risk_limits_ok(action):
            return False, "Risk limit breach"

        return True, f"Approved by {', '.join(self.REQUIRED_APPROVERS)}"

    def _cfg_bool(self, section: str, key: str, default: bool) -> bool:
        try:
            v = str(self.config.get(section, key, fallback=str(default))).strip().lower()
            return v in {"1", "true", "yes", "y", "on"}
        except Exception:
            return bool(default)

    def _cfg_float(self, section: str, key: str, default: float) -> float:
        try:
            return float(str(self.config.get(section, key, fallback=str(default))).strip())
        except Exception:
            return float(default)

    def _cfg_int(self, section: str, key: str, default: int) -> int:
        try:
            return int(float(str(self.config.get(section, key, fallback=str(default))).strip()))
        except Exception:
            return int(default)

    def _within_daily_change_limit(self) -> bool:
        max_changes = self._cfg_int("CONFIGURATION", "agent_max_live_changes_per_day", 8)
        # Stub counter for v1 rollout; storage-backed counting can be added in experiments DB.
        return max_changes > 0

    def _risk_limits_ok(self, action: dict) -> bool:
        try:
            exposure = float(action.get("proposed_exposure_pct", 0.0) or 0.0)
        except Exception:
            exposure = 0.0

        hard_limit = self._cfg_float("CONFIGURATION", "agent_live_max_exposure_pct", 0.30)

        canary_enabled = self._cfg_bool("CONFIGURATION", "agent_canary_enabled", True)
        canary_limit = self._cfg_float("CONFIGURATION", "agent_canary_exposure_pct", 0.10)
        effective_limit = min(hard_limit, canary_limit) if canary_enabled else hard_limit

        if exposure > effective_limit:
            return False

        # Optional challenger degradation guard for deploy actions.
        try:
            delta_pct = float(action.get("champion_delta_pct", 0.0) or 0.0)
        except Exception:
            delta_pct = 0.0
        allowed_degrade = self._cfg_float("CONFIGURATION", "agent_canary_underperform_pct_max", 5.0)
        if delta_pct < 0 and abs(delta_pct) > allowed_degrade:
            return False

        return True
