import configparser
from datetime import datetime, timedelta, timezone
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
        self.metrics = []
        self.anomalies = []

    def log_metric(self, name, value, tags=None):
        self.metrics.append((name, value, tags or {}))

    def log_anomaly(self, code, severity="WARN", source="", details=None):
        self.anomalies.append((code, severity, source, details or {}))


class _DummyDB:
    def __init__(self, ts_map):
        self.ts_map = dict(ts_map)

    def get_last_timestamp(self, symbol):
        return self.ts_map.get(symbol)


class _DummyAgent:
    def __init__(self, cfg, db):
        self.config = cfg
        self.db = db
        self.metrics = _DummyMetrics()
        self._agent_started_at = 0.0
        self._persist_calls = 0
        self._backfill_calls = 0
        self.logs = []

    def _cfg_int(self, section, key, default=0):
        try:
            return int(float(self.config.get(section, key, fallback=str(default))))
        except Exception:
            return int(default)

    def _cfg_bool(self, section, key, default=False):
        v = self.config.get(section, key, fallback=str(default))
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    def _persist_split_config(self):
        self._persist_calls += 1

    def _run_auto_backfill(self):
        self._backfill_calls += 1

    def log(self, msg):
        self.logs.append(str(msg))


def _cfg_with_active(*symbols):
    cfg = configparser.ConfigParser()
    cfg["CONFIGURATION"] = {
        "agent_stale_quarantine_threshold_seconds": "3600",
        "agent_stale_quarantine_warmup_minutes": "0",
        "agent_auto_backfill_enabled": "True",
    }
    cfg["WATCHLIST_ACTIVE_STOCK"] = {s: "" for s in symbols if "/" not in s}
    cfg["WATCHLIST_ACTIVE_CRYPTO"] = {s: "" for s in symbols if "/" in s}
    cfg["WATCHLIST_ARCHIVE_STOCK"] = {}
    cfg["WATCHLIST_ARCHIVE_CRYPTO"] = {}
    return cfg


def test_uninitialized_symbols_are_not_quarantined_and_trigger_backfill():
    cfg = _cfg_with_active("AAPL")
    agent = _DummyAgent(cfg, _DummyDB({"AAPL": None}))

    AgentMaster._run_stale_symbol_quarantine(agent)

    assert "AAPL" in cfg["WATCHLIST_ACTIVE_STOCK"]
    assert "AAPL" not in cfg["WATCHLIST_ARCHIVE_STOCK"]
    assert agent._backfill_calls == 1
    assert agent._persist_calls == 0


def test_warmup_window_disables_quarantine():
    cfg = _cfg_with_active("AAPL")
    cfg["CONFIGURATION"]["agent_stale_quarantine_warmup_minutes"] = "60"
    old_ts = datetime.now(timezone.utc) - timedelta(hours=5)
    agent = _DummyAgent(cfg, _DummyDB({"AAPL": old_ts}))
    agent._agent_started_at = datetime.now(timezone.utc).timestamp()

    AgentMaster._run_stale_symbol_quarantine(agent)

    assert "AAPL" in cfg["WATCHLIST_ACTIVE_STOCK"]
    assert "AAPL" not in cfg["WATCHLIST_ARCHIVE_STOCK"]
    assert any(m[0] == "agent_stale_quarantine_warmup_remaining_seconds" for m in agent.metrics.metrics)


def test_stale_symbols_with_history_are_quarantined_after_warmup():
    cfg = _cfg_with_active("AAPL")
    old_ts = datetime.now(timezone.utc) - timedelta(hours=5)
    agent = _DummyAgent(cfg, _DummyDB({"AAPL": old_ts}))
    agent._agent_started_at = 0.0

    AgentMaster._run_stale_symbol_quarantine(agent)

    assert "AAPL" not in cfg["WATCHLIST_ACTIVE_STOCK"]
    assert "AAPL" in cfg["WATCHLIST_ARCHIVE_STOCK"]
    assert agent._persist_calls == 1
    assert any(a[0] == "STALE_SYMBOL_QUARANTINE" for a in agent.metrics.anomalies)
