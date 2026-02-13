from modules.tests.engine_test_stubs import import_engine_core_with_stubs


def _make_engine(core_module):
    engine = core_module.TradingEngine.__new__(core_module.TradingEngine)
    engine._min_log_level = 0
    engine._log_throttle = {}
    engine._publish_agent_event = lambda *_args, **_kwargs: None
    engine._parse_level = lambda _lvl, default=20: 20
    return engine


def test_emit_forwards_order_context_to_structured_logger(monkeypatch):
    core = import_engine_core_with_stubs(monkeypatch)
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
    core = import_engine_core_with_stubs(monkeypatch)
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
