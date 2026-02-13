from modules.tests.engine_test_stubs import import_engine_core_with_stubs


class _DummyDB:
    def log_trade_entry(self, *_args, **_kwargs):
        raise AssertionError("TTL cancellation path should not log a filled trade")


class _DummyAPI:
    def __init__(self):
        self.canceled = []

    def get_position(self, _symbol):
        raise RuntimeError("no position")

    def get_order(self, _oid):
        return None

    def cancel_order(self, oid):
        self.canceled.append(oid)


def _make_engine(core_module):
    engine = core_module.TradingEngine.__new__(core_module.TradingEngine)
    engine.api = _DummyAPI()
    engine.db = _DummyDB()
    engine.pending_orders = {
        "oid-1": {
            "symbol": "AAPL",
            "strategy": "TEST",
            "qty": 1,
            "price": 100.0,
            "submitted_ts": 1000.0,
        }
    }
    engine._pending_symbols = {"AAPL"}
    engine._live_entry_ttl_sec = 60
    engine._live_cancel_unfilled_entries = True
    engine._log_order_lifecycle = True
    engine._order_stats = {
        "submitted": 0,
        "filled": 0,
        "canceled": 0,
        "rejected": 0,
        "expired": 0,
        "ttl_canceled": 0,
    }
    engine._e5_ttl_cancel_events = []
    engine.retry_api_call = lambda fn, *a, **k: fn(*a, **k)
    engine._redact = lambda e: str(e)
    engine._log_exec_packet = lambda **_kwargs: None
    engine._emit_calls = []
    engine._emit = lambda *args, **kwargs: engine._emit_calls.append({"args": args, "kwargs": kwargs})
    engine._e5_note_ttl_cancel = lambda _symbol=None: engine._e5_ttl_cancel_events.append(_symbol)
    return engine


def test_process_pending_orders_cancels_unfilled_entry_after_ttl(monkeypatch):
    core = import_engine_core_with_stubs(monkeypatch)
    engine = _make_engine(core)
    monkeypatch.setattr(core.time, "time", lambda: 1100.0)

    engine.process_pending_orders()

    assert engine.api.canceled == ["oid-1"]
    assert engine.pending_orders == {}
    assert "AAPL" not in engine._pending_symbols
    assert engine._order_stats["ttl_canceled"] == 1
    assert engine._order_stats["canceled"] == 1
    assert engine._e5_ttl_cancel_events == ["AAPL"]
    ttl_logs = [c["kwargs"] for c in engine._emit_calls if "CANCELED unfilled BUY" in str(c["args"][0])]
    assert len(ttl_logs) == 1
    assert ttl_logs[0]["order_id"] == "oid-1"
    assert ttl_logs[0]["strategy"] == "TEST"


def test_process_pending_orders_does_not_cancel_when_ttl_disabled(monkeypatch):
    core = import_engine_core_with_stubs(monkeypatch)
    engine = _make_engine(core)
    engine._live_cancel_unfilled_entries = False
    monkeypatch.setattr(core.time, "time", lambda: 1100.0)

    engine.process_pending_orders()

    assert engine.api.canceled == []
    assert "oid-1" in engine.pending_orders
    assert engine._order_stats["ttl_canceled"] == 0