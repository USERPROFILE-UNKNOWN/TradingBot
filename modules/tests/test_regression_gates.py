import configparser
import importlib
import sys
import types
from pathlib import Path


def _load_agent_master_with_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    sys.modules.pop("modules.agent_master", None)
    mod = importlib.import_module("modules.agent_master")
    return importlib.reload(mod)


def _load_engine_core_with_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
    monkeypatch.setitem(sys.modules, "pandas_ta", types.ModuleType("pandas_ta"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))

    fake_requests = types.ModuleType("requests")

    class _ReqExc:
        ConnectionError = Exception
        ReadTimeout = Exception

    fake_requests.exceptions = _ReqExc
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    fake_strategies = types.ModuleType("modules.strategies")

    class _DummyOptimizer:
        def __init__(self, *_args, **_kwargs):
            return None

    class _DummyWallet:
        def __init__(self, *_args, **_kwargs):
            return None

    fake_strategies.StrategyOptimizer = _DummyOptimizer
    fake_strategies.WalletManager = _DummyWallet
    monkeypatch.setitem(sys.modules, "modules.strategies", fake_strategies)

    fake_sentiment = types.ModuleType("modules.sentiment")

    class _DummySentinel:
        def __init__(self, *_args, **_kwargs):
            return None

    fake_sentiment.NewsSentinel = _DummySentinel
    monkeypatch.setitem(sys.modules, "modules.sentiment", fake_sentiment)

    sys.modules.pop("modules.engine.core", None)
    sys.modules.pop("modules.engine", None)
    core = importlib.import_module("modules.engine.core")
    return importlib.reload(core)


def test_agentmaster_init_does_not_start_scheduler(monkeypatch):
    mod = _load_agent_master_with_stubs(monkeypatch)

    calls = {"started": 0}

    class _DummyScheduler:
        def __init__(self, *args, **kwargs):
            return None

        def start(self):
            calls["started"] += 1

    monkeypatch.setattr(mod, "JobScheduler", _DummyScheduler, raising=True)

    cfg = configparser.ConfigParser()
    cfg["CONFIGURATION"] = {"agent_mode": "OFF"}
    cfg["TRADINGVIEW"] = {"enabled": "False"}

    class _DB:
        db_dir = "db"

    mod.AgentMaster(cfg, _DB(), log_callback=lambda *_a, **_k: None)
    assert calls["started"] == 0


def test_engine_boot_uses_disabled_ai_when_ai_module_unavailable(monkeypatch):
    core = _load_engine_core_with_stubs(monkeypatch)

    real_import_module = importlib.import_module

    def _fake_import(name, package=None):
        if name == "modules.ai":
            raise ImportError("simulated missing optional ai")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import, raising=True)

    class _DB:
        pass

    cfg = configparser.ConfigParser()
    cfg["CONFIGURATION"] = {
        "amount_to_trade": "2000",
        "max_daily_loss": "100",
        "equity_data_feed": "iex",
        "ai_runtime_training_enabled": "False",
    }

    e = core.TradingEngine(cfg, _DB(), lambda *_a, **_k: None, lambda *_a, **_k: None)
    assert type(e.ai).__name__ == "_DisabledAIOracle"
    assert float(e.ai.predict_probability({})) == 0.5


def test_config_regression_keys_and_no_auto_added_defaults_banner():
    root = Path(__file__).resolve().parents[2]
    config_ini = (root / "config" / "config.ini").read_text(encoding="utf-8")
    defaults_py = (root / "modules" / "config_defaults.py").read_text(encoding="utf-8")

    assert "AUTO-ADDED DEFAULTS (preserve comments)" not in config_ini
    for key in (
        "agent_autopilot_enabled",
        "agent_autopilot_require_engine_running_for_mutations",
        "ai_runtime_training_enabled",
        "watchlist_auto_update_enabled",
    ):
        assert key in config_ini
        assert key in defaults_py
