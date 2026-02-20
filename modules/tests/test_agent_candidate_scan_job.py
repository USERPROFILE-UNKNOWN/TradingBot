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


class _DummyAgent:
    def __init__(self, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self._logger = object()
        self.db = object()
        self.config = configparser.ConfigParser()
        self.metrics = _DummyMetrics()
        self.logs = []
        self.published = []

    def log(self, msg):
        self.logs.append(str(msg))

    def publish(self, event_type, payload=None):
        self.published.append((event_type, payload or {}))


def test_candidate_scan_job_runs_and_publishes(monkeypatch):
    rows = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]

    class _Scanner:
        def __init__(self, _db, _cfg, log=None):
            return None

        def scan_today(self):
            return "scan-1", rows

    fake_mod = types.ModuleType("modules.research.candidate_scanner")
    fake_mod.CandidateScanner = _Scanner
    monkeypatch.setitem(sys.modules, "modules.research.candidate_scanner", fake_mod)

    a = _DummyAgent(mode="PAPER")
    AgentMaster._run_candidate_scan(a)

    assert a.metrics.started
    assert a.metrics.finished
    assert any(m[0] == "agent_candidate_scan_rows" and m[1] == 2.0 for m in a.metrics.metrics)
    assert any(ev[0] == "candidate_scan_completed" and ev[1].get("count") == 2 for ev in a.published)


def test_candidate_scan_job_skips_outside_paper_live(monkeypatch):
    called = {"scanner": False}

    class _Scanner:
        def __init__(self, _db, _cfg, log=None):
            called["scanner"] = True

        def scan_today(self):
            return "scan-1", []

    fake_mod = types.ModuleType("modules.research.candidate_scanner")
    fake_mod.CandidateScanner = _Scanner
    monkeypatch.setitem(sys.modules, "modules.research.candidate_scanner", fake_mod)

    a = _DummyAgent(mode="OFF")
    AgentMaster._run_candidate_scan(a)

    assert called["scanner"] is False
    assert not a.metrics.started
    assert not a.published
