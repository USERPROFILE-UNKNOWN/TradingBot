import configparser
import importlib
import sys
import types


def _load_agent_master_module():
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    return importlib.import_module("modules.agent_master")


def _cfg(enabled=True):
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {
        "agent_mode": "PAPER",
        "agent_shadow_enabled": "True" if enabled else "False",
        "agent_shadow_interval_sec": "300",
        "agent_shadow_include_summaries": "True",
        "agent_shadow_include_backtests": "True",
    }
    return c


class _DummyDB:
    def __init__(self):
        self.paths = {"logs": "./logs"}


def test_start_agent_shadow_if_enabled_starts_shadow(monkeypatch):
    m = _load_agent_master_module()
    AgentMaster = m.AgentMaster

    started = {"value": 0}

    class _Shadow:
        def __init__(self, *_a, **_k):
            return None

        def start(self):
            started["value"] += 1

        def stop(self):
            return None

    monkeypatch.setattr(m, "AgentShadow", _Shadow, raising=True)

    class _DummyAgent:
        def __init__(self):
            self.config = _cfg(enabled=True)
            self.db = _DummyDB()
            self.logs = []
            self._shadow_agent = None

        def _cfg_bool(self, section, key, default=False):
            v = self.config.get(section, key, fallback=str(default))
            return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

        def _resolve_runtime_paths(self):
            return {"logs": "./logs"}

        def log(self, msg):
            self.logs.append(str(msg))

    a = _DummyAgent()
    AgentMaster._start_agent_shadow_if_enabled(a)

    assert started["value"] == 1
    assert a._shadow_agent is not None


def test_start_agent_shadow_if_disabled_is_noop(monkeypatch):
    m = _load_agent_master_module()
    AgentMaster = m.AgentMaster

    started = {"value": 0}

    class _Shadow:
        def __init__(self, *_a, **_k):
            return None

        def start(self):
            started["value"] += 1

        def stop(self):
            return None

    monkeypatch.setattr(m, "AgentShadow", _Shadow, raising=True)

    class _DummyAgent:
        def __init__(self):
            self.config = _cfg(enabled=False)
            self.db = _DummyDB()
            self.logs = []
            self._shadow_agent = None

        def _cfg_bool(self, section, key, default=False):
            v = self.config.get(section, key, fallback=str(default))
            return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

        def _resolve_runtime_paths(self):
            return {"logs": "./logs"}

        def log(self, msg):
            self.logs.append(str(msg))

    a = _DummyAgent()
    AgentMaster._start_agent_shadow_if_enabled(a)

    assert started["value"] == 0
    assert a._shadow_agent is None
