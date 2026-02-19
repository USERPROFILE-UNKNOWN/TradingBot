import configparser
import importlib
import sys
import types
import threading


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
        self.anoms = []

    def start_autopilot_run(self, run_id, mode, phase, status="OK", summary=None):
        self.started.append((run_id, mode, phase, status, summary or {}))

    def finish_autopilot_run(self, run_id, status="OK", summary=None):
        self.finished.append((run_id, status, summary or {}))

    def log_anomaly(self, event_type, severity="WARN", source="", details=None):
        self.anoms.append((event_type, severity, source, details or {}))


class _DummyAgent:
    def __init__(self, mode="PAPER"):
        self.mode = mode
        self._autopilot_seq = 0
        self.config = configparser.ConfigParser()
        self.config["CONFIGURATION"] = {
            "agent_watchlist_policy_max_churn_per_run": "1",
            "agent_autopilot_require_engine_running_for_mutations": "True",
        }
        self.config["WATCHLIST_ACTIVE_STOCK"] = {"AAPL": ""}
        self.config["WATCHLIST_ACTIVE_CRYPTO"] = {}
        self.config["WATCHLIST_ARCHIVE_STOCK"] = {}
        self.config["WATCHLIST_ARCHIVE_CRYPTO"] = {}
        self.db = types.SimpleNamespace(paths={})
        self.metrics = _DummyMetrics()
        self.published = []
        self._state_write_lock = threading.RLock()
        self._engine_running_flag = True

    def _cfg_int(self, section, key, default=0):
        return int(float(self.config.get(section, key, fallback=str(default))))

    def _cfg_bool(self, section, key, default=False):
        v = self.config.get(section, key, fallback=str(default))
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    def _cfg_str(self, section, key, default=""):
        return str(self.config.get(section, key, fallback=default))

    def publish(self, ev, payload=None):
        self.published.append((ev, payload or {}))

    def _engine_running(self):
        return bool(self._engine_running_flag)

    def _persist_split_config(self):
        return None

    def log(self, _msg):
        return None


def test_watchlist_policy_job_runs_and_logs_churn_warning(monkeypatch):
    def _fake_apply(*_a, **_k):
        return {
            "batch_id": "b1",
            "added": ["MSFT"],
            "removed": ["AAPL"],
            "new_watchlist": ["MSFT"],
            "rejected": {"AAPL": ["already_active"]},
        }

    monkeypatch.setattr("modules.agent_master.apply_watchlist_policy", _fake_apply, raising=True)

    a = _DummyAgent()
    AgentMaster._run_watchlist_policy_update(a)

    assert a.metrics.started
    assert a.metrics.finished
    assert any(ev[0] == "watchlist_policy_updated" for ev in a.published)
    assert any(x[0] == "WATCHLIST_POLICY_CHURN_EXCEEDED" for x in a.metrics.anoms)


def test_watchlist_policy_job_skips_mutation_when_engine_stopped(monkeypatch):
    called = {"n": 0}

    def _fake_apply(*_a, **_k):
        called["n"] += 1
        return {}

    monkeypatch.setattr("modules.agent_master.apply_watchlist_policy", _fake_apply, raising=True)

    a = _DummyAgent()
    a._engine_running_flag = False
    AgentMaster._run_watchlist_policy_update(a)

    assert called["n"] == 0
    assert any(x[0] == "AUTOPILOT_MUTATION_BLOCKED" for x in a.metrics.anoms)
    assert a.metrics.finished
    assert a.metrics.finished[-1][1] == "SKIP"
