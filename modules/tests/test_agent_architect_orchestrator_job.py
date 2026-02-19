import configparser
import importlib
import sys
import types
from datetime import datetime, timezone


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


class _DummyDB:
    def __init__(self):
        self.paths = {"logs": "logs"}
        self.queue = [
            {"id": "ARCH001", "source_symbol": "AAPL", "genome": {"rsi": 30, "sl": 0.02, "tp": 2.0, "ema": True}},
            {"id": "ARCH002", "source_symbol": "MSFT", "genome": {"rsi": 35, "sl": 0.03, "tp": 2.5, "ema": False}},
        ]
        self.marked = []

    def architect_queue_list(self, statuses=None, limit=10):
        assert statuses == ["NEW"]
        return list(self.queue[:limit])

    def architect_queue_mark_status(self, item_id, status, result_pointer="", error_text=""):
        self.marked.append((item_id, status, result_pointer, error_text))
        return True


class _DummyAgent:
    def __init__(self, db, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self._logger = object()
        self.db = db
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {
            "agent_architect_orchestrator_hour_utc": str(datetime.now(timezone.utc).hour),
            "agent_architect_orchestrator_max_queue_items": "5",
            "agent_architect_orchestrator_max_symbols": "4",
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


def test_architect_orchestrator_processes_queue(monkeypatch):
    fake_runner = types.ModuleType("modules.backtest_runner")

    def _normalize(items):
        return [types.SimpleNamespace(vid=it["id"], source_symbol=it["source_symbol"], genome=it["genome"], notes={}) for it in items]

    def _run(**kwargs):
        assert kwargs["symbols"] == ["AAPL", "MSFT"]
        return {"df": object(), "aggregates": {}, "variants": [], "meta": {}}

    fake_runner.normalize_variants = _normalize
    fake_runner.run_architect_queue_backtest = _run
    monkeypatch.setitem(sys.modules, "modules.backtest_runner", fake_runner)

    fake_export = types.ModuleType("modules.research.architect_backtest_exporter")
    fake_export.export_architect_backtest_bundle = lambda **_k: {"json": "logs/backtest/bundle.json"}
    monkeypatch.setitem(sys.modules, "modules.research.architect_backtest_exporter", fake_export)

    fake_utils = types.ModuleType("modules.utils")
    fake_utils.APP_VERSION = "vX"
    fake_utils.APP_RELEASE = "rel"
    monkeypatch.setitem(sys.modules, "modules.utils", fake_utils)

    mod = importlib.import_module("modules.agent_master")
    monkeypatch.setattr(mod, "get_watchlist_symbols", lambda *_a, **_k: ["AAPL", "MSFT"])

    db = _DummyDB()
    a = _DummyAgent(db)
    AgentMaster._run_architect_orchestrator(a)

    assert a.metrics.started
    assert a.metrics.finished
    assert any(m[0] == "agent_architect_orchestrator_processed" and m[1] == 2.0 for m in a.metrics.metrics)
    assert any(ev[0] == "architect_orchestrator_completed" and ev[1].get("processed") == 2 for ev in a.published)
    assert len([m for m in db.marked if m[1] == "RUNNING"]) == 2
    assert len([m for m in db.marked if m[1] == "DONE"]) == 2


def test_architect_orchestrator_skips_non_live_modes():
    db = _DummyDB()
    a = _DummyAgent(db, mode="OFF")
    AgentMaster._run_architect_orchestrator(a)

    assert not a.metrics.started
    assert not db.marked
