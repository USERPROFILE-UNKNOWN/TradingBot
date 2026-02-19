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
        self.enqueued = []

    def get_latest_candidates(self, limit=10):
        return _Rows(self._rows[:limit])

    def architect_queue_enqueue(self, source_symbol, genome, metrics=None):
        self.enqueued.append((source_symbol, dict(genome), dict(metrics or {})))
        return f"ARCH{len(self.enqueued):03d}"


class _DummyAgent:
    def __init__(self, db, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self._logger = object()
        self.db = db
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {
            "agent_architect_optimize_max_symbols": "2",
            "agent_architect_optimize_top_variants_per_symbol": "2",
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


def test_architect_optimize_enqueues_top_variants(monkeypatch):
    class _StubArchitect:
        def __init__(self, _db, config=None):
            self.config = config

        def run_optimization(self, symbol, _progress_callback):
            return [
                {"rsi": 30, "sl": 0.02, "tp": 2.0, "ema": True, "score": 10, "profit": 5.0, "trades": 3, "win_rate": 66.0},
                {"rsi": 35, "sl": 0.03, "tp": 2.5, "ema": False, "score": 8, "profit": 4.0, "trades": 2, "win_rate": 50.0},
                {"rsi": 40, "sl": 0.04, "tp": 3.0, "ema": True, "score": 1, "profit": 1.0, "trades": 1, "win_rate": 25.0},
            ]

    fake_mod = types.ModuleType("modules.architect")
    fake_mod.TheArchitect = _StubArchitect
    monkeypatch.setitem(sys.modules, "modules.architect", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "NVDA"}])
    a = _DummyAgent(db)
    AgentMaster._run_architect_optimize(a)

    assert a.metrics.started
    assert a.metrics.finished
    assert len(db.enqueued) == 4  # 2 symbols * top 2 variants
    assert any(m[0] == "agent_architect_optimize_queued" and m[1] == 4.0 for m in a.metrics.metrics)
    assert any(ev[0] == "architect_optimize_completed" and ev[1].get("queued") == 4 for ev in a.published)


def test_architect_optimize_skips_non_live_modes(monkeypatch):
    class _StubArchitect:
        def __init__(self, *_a, **_k):
            pass

        def run_optimization(self, *_a, **_k):
            raise AssertionError("should not run")

    fake_mod = types.ModuleType("modules.architect")
    fake_mod.TheArchitect = _StubArchitect
    monkeypatch.setitem(sys.modules, "modules.architect", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}])
    a = _DummyAgent(db, mode="OFF")
    AgentMaster._run_architect_optimize(a)

    assert not a.metrics.started
    assert not db.enqueued
