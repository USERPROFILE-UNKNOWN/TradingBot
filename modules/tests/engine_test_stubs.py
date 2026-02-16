import importlib
import sys
import types


def import_engine_core_with_stubs(monkeypatch):
    """Import modules.engine.core with lightweight dependency stubs."""
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
    monkeypatch.setitem(sys.modules, "pandas_ta", types.ModuleType("pandas_ta"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))

    # Minimal requests.exceptions surface so retry_api_call() can execute safely.
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

    fake_ai = types.ModuleType("modules.ai")

    class _DummyOracle:
        def __init__(self, *_args, **_kwargs):
            return None

        # TradingEngine spawns a daemon thread that calls train_model().
        # Keep it a no-op to avoid noisy thread exceptions during tests.
        def train_model(self):
            return None

    fake_ai.AI_Oracle = _DummyOracle
    monkeypatch.setitem(sys.modules, "modules.ai", fake_ai)

    # Ensure a clean import so tests do not depend on prior module state.
    sys.modules.pop("modules.engine.core", None)
    sys.modules.pop("modules.engine", None)

    core = importlib.import_module("modules.engine.core")
    return importlib.reload(core)
