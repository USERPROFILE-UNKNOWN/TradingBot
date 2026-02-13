import importlib
import sys
import types


def _import_engine_core_with_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
    monkeypatch.setitem(sys.modules, "pandas_ta", types.ModuleType("pandas_ta"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))

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

    fake_ai.AI_Oracle = _DummyOracle
    monkeypatch.setitem(sys.modules, "modules.ai", fake_ai)

    core = importlib.import_module("modules.engine.core")
    return importlib.reload(core)


def _make_engine(core_module):
    engine = core_module.TradingEngine.__new__(core_module.TradingEngine)
    engine._min_log_level = 0
    engine._log_throttle = {}
    engine._publish_agent_event = lambda *_args, **_kwargs: None
    engine._parse_level = lambda _lvl, default=20: 20
    return engine


def test_emit_forwards_order_context_to_structured_logger(monkeypatch):
    core = _import_engine_core_with_stubs(monkeypatch)
    engine = _make_engine(core)
    calls = []

    def _log(msg, **kwargs):
        calls.append((msg, kwargs))

    engine.log = _log

    engine._emit(
        "filled",
        level="INFO",
        category="ORDER",
        symbol="AAPL",
        order_id="oid-7",
        strategy="MOMO",
    )

    assert len(calls) == 1
    assert calls[0][0] == "filled"
    assert calls[0][1]["order_id"] == "oid-7"
    assert calls[0][1]["strategy"] == "MOMO"


def test_emit_fallback_prefix_includes_order_context_when_logger_is_legacy(monkeypatch):
    core = _import_engine_core_with_stubs(monkeypatch)
    engine = _make_engine(core)
    messages = []

    def _legacy_log(msg):
        messages.append(msg)

    engine.log = _legacy_log

    engine._emit(
        "ttl cancel",
        level="WARN",
        category="ORDER",
        symbol="AAPL",
        order_id="oid-9",
        strategy="MEANREV",
    )

    assert len(messages) == 1
    out = messages[0]
    assert "[WARN]" in out
    assert "[ORDER]" in out
    assert "[AAPL]" in out
    assert "[OID:oid-9]" in out
    assert "[STRAT:MEANREV]" in out
