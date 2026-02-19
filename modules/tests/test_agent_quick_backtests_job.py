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
        self.started = []
        self.finished = []
        self.metrics = []
        self.anoms = []

    def start_autopilot_run(self, run_id, mode, phase, status="OK", summary=None):
        self.started.append((run_id, mode, phase, status, summary or {}))

    def finish_autopilot_run(self, run_id, status="OK", summary=None):
        self.finished.append((run_id, status, summary or {}))

    def log_metric(self, name, value, metadata=None):
        self.metrics.append((name, value, metadata or {}))

    def log_anomaly(self, event_type, severity="WARN", source="", details=None):
        self.anoms.append((event_type, severity, source, details or {}))


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return list(self._rows)


class _DummyDB:
    def __init__(self, rows):
        self._rows = rows
        self.saved = []

    def get_latest_candidates(self, limit=10):
        return _Rows(self._rows[:limit])

    def save_backtest_result(self, data):
        self.saved.append(dict(data))


class _DummyAgent:
    def __init__(self, db, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self._logger = object()
        self.db = db
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {
            "agent_quick_backtest_max_symbols": "5",
            "agent_quick_backtest_days": "14",
            "agent_quick_backtest_max_strategies": "6",
            "agent_quick_backtest_min_trades": "1",
        }
        self.metrics = _DummyMetrics()
        self.published = []
        self.logs = []

    def _cfg_int(self, section, key, default=0):
        try:
            return int(float(self.config.get(section, key, fallback=str(default))))
        except Exception:
            return int(default)

    def publish(self, event_type, payload=None):
        self.published.append((event_type, payload or {}))

    def log(self, msg):
        self.logs.append(str(msg))


def test_quick_backtests_job_runs_and_persists(monkeypatch):
    def _fake_qb(_cfg, _db, symbol, days=14, max_strategies=6, min_trades=1):
        return {
            "ok": True,
            "symbol": symbol,
            "best": {
                "strategy": "STRAT_A",
                "total_pl": 12.3,
                "win_rate": 0.6,
                "max_drawdown": -1.1,
                "trades": 4,
            },
        }

    fake_mod = types.ModuleType("modules.research.quick_backtest")
    fake_mod.run_quick_backtest = _fake_qb
    monkeypatch.setitem(sys.modules, "modules.research.quick_backtest", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}, {"symbol": "MSFT"}])
    a = _DummyAgent(db)
    AgentMaster._run_quick_backtests(a)

    assert a.metrics.started
    assert a.metrics.finished
    assert len(db.saved) == 2
    assert any(m[0] == "agent_quick_backtest_symbols" and m[1] == 2.0 for m in a.metrics.metrics)
    assert any(ev[0] == "quick_backtests_completed" and ev[1].get("ok") == 2 for ev in a.published)


def test_quick_backtests_job_skips_non_live_modes(monkeypatch):
    called = {"qb": False}

    def _fake_qb(*_a, **_k):
        called["qb"] = True
        return {"ok": False}

    fake_mod = types.ModuleType("modules.research.quick_backtest")
    fake_mod.run_quick_backtest = _fake_qb
    monkeypatch.setitem(sys.modules, "modules.research.quick_backtest", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}])
    a = _DummyAgent(db, mode="OFF")
    AgentMaster._run_quick_backtests(a)

    assert called["qb"] is False
    assert not a.metrics.started
    assert not db.saved
