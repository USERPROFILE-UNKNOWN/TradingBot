import configparser
import importlib
import sys
import types


def _load_agent_master():
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    m = importlib.import_module("modules.agent_master")
    return m.AgentMaster


AgentMaster = _load_agent_master()


class _DummyMetrics:
    def __init__(self):
        self.actions = []

    def log_agent_action(self, mode, action_type, approved, reason, action):
        self.actions.append((mode, action_type, approved, reason, dict(action or {})))


class _DummyGov:
    def approve_action(self, _action):
        return True, "Approved"


class _DummyAgent:
    def __init__(self):
        self.mode = "LIVE"
        self._hard_halt_active = False
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {
            "promotion_enabled": "True",
            "promotion_min_sessions": "5",
            "promotion_min_trades_total": "30",
            "promotion_max_drawdown_pct": "4.0",
            "promotion_max_cancel_rate_pct": "35.0",
            "promotion_max_reject_rate_pct": "10.0",
            "promotion_require_no_watchdog_halts": "True",
            "promotion_require_no_stale_symbols": "True",
            "agent_hard_halt_supreme": "True",
            "agent_max_config_tunes_per_day": "2",
            "agent_max_promotions_per_day": "2",
            "agent_promotion_enabled": "True",
            "agent_live_max_exposure_pct": "0.30",
            "agent_canary_enabled": "False",
            "agent_canary_exposure_pct": "0.10",
            "agent_canary_underperform_pct_max": "5.0",
        }
        self.metrics = _DummyMetrics()
        self.gov = _DummyGov()
        self._live_change_day = ""
        self._live_config_tunes_today = 0
        self._live_promotions_today = 0

    def _cfg_bool(self, section, key, default=False):
        try:
            v = str(self.config.get(section, key, fallback=str(default))).strip().lower()
            return v in {"1", "true", "yes", "y", "on"}
        except Exception:
            return bool(default)

    def _cfg_int(self, section, key, default=0):
        try:
            return int(float(self.config.get(section, key, fallback=str(default))))
        except Exception:
            return int(default)

    def _cfg_float(self, section, key, default=0.0):
        try:
            return float(self.config.get(section, key, fallback=str(default)))
        except Exception:
            return float(default)

    def _check_live_promotion_gates(self, action):
        return AgentMaster._check_live_promotion_gates(self, action)


def test_live_deploy_requires_promotion_gates_pass():
    a = _DummyAgent()
    action = {
        "type": "DEPLOY_STRATEGY",
        "proposed_exposure_pct": 0.05,
        "promotion_stats": {
            "paper_sessions": 6,
            "paper_trades_total": 40,
            "drawdown_pct": 2.0,
            "cancel_rate_pct": 5.0,
            "reject_rate_pct": 1.0,
            "watchdog_halts": 0,
            "stale_symbols": 0,
        },
    }
    ok, reason = AgentMaster.evaluate_action(a, action)
    assert ok is True
    assert "Approved" in reason


def test_live_deploy_blocked_when_promotion_gates_fail():
    a = _DummyAgent()
    action = {
        "type": "DEPLOY_STRATEGY",
        "proposed_exposure_pct": 0.05,
        "promotion_stats": {
            "paper_sessions": 2,
            "paper_trades_total": 10,
            "drawdown_pct": 7.0,
            "cancel_rate_pct": 40.0,
            "reject_rate_pct": 20.0,
            "watchdog_halts": 1,
            "stale_symbols": 1,
        },
    }
    ok, reason = AgentMaster.evaluate_action(a, action)
    assert ok is False
    assert "Promotion gate failed" in reason
