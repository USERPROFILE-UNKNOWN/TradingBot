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

    def save_candidates(self, scan_id, rows, universe="AUTO", policy="AUTO", scan_ts=None):
        self.saved.append((scan_id, rows, universe, policy))


class _DummyAgent:
    def __init__(self, db, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self._logger = object()
        self.db = db
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {"agent_candidate_simulation_max_symbols": "5"}
        self.metrics = _DummyMetrics()
        self.logs = []
        self.published = []

    def log(self, msg):
        self.logs.append(str(msg))

    def publish(self, event_type, payload=None):
        self.published.append((event_type, payload or {}))

    def _cfg_int(self, section, key, default=0):
        try:
            return int(float(self.config.get(section, key, fallback=str(default))))
        except Exception:
            return int(default)


def test_candidate_simulation_job_scores_and_persists(monkeypatch):
    class _Scanner:
        def __init__(self, _db, _cfg, log=None):
            return None

        def score_single_symbol(self, symbol, **_kwargs):
            return {"symbol": symbol, "score": 42.0, "universe": "SIMULATION", "reason": "ok"}

    fake_mod = types.ModuleType("modules.research.candidate_scanner")
    fake_mod.CandidateScanner = _Scanner
    monkeypatch.setitem(sys.modules, "modules.research.candidate_scanner", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}, {"symbol": "MSFT"}])
    a = _DummyAgent(db)
    AgentMaster._run_candidate_simulation(a, payload={"scan_id": "scan-abc"})

    assert a.metrics.started
    assert a.metrics.finished
    assert any(m[0] == "agent_candidate_simulation_rows" and m[1] == 2.0 for m in a.metrics.metrics)
    assert any(ev[0] == "candidate_simulation_completed" and ev[1].get("count") == 2 for ev in a.published)
    assert db.saved and db.saved[0][2] == "SIMULATION"


def test_candidate_simulation_job_skips_non_live_modes(monkeypatch):
    called = {"scanner": False}

    class _Scanner:
        def __init__(self, _db, _cfg, log=None):
            called["scanner"] = True

    fake_mod = types.ModuleType("modules.research.candidate_scanner")
    fake_mod.CandidateScanner = _Scanner
    monkeypatch.setitem(sys.modules, "modules.research.candidate_scanner", fake_mod)

    db = _DummyDB(rows=[{"symbol": "AAPL"}])
    a = _DummyAgent(db, mode="OFF")
    AgentMaster._run_candidate_simulation(a)

    assert called["scanner"] is False
    assert not a.metrics.started
