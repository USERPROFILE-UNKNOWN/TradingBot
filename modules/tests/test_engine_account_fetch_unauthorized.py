import configparser

from modules.tests.engine_test_stubs import import_engine_core_with_stubs


class _DummyDB:
    def get_active_trades(self):
        return {}


class _DummyGateway:
    def __init__(self, *_args, **_kwargs):
        self.base_url = "https://paper-api.alpaca.markets"

    def connect(self):
        return None

    def retry_api_call(self, *_args, **_kwargs):
        # Simulate account fetch/list_positions returning None after unauthorized errors.
        return None

    def get_account(self):
        return None

    def list_positions(self):
        return []


def test_account_fetch_failure_reports_auth_error_and_skips_sync(monkeypatch):
    core = import_engine_core_with_stubs(monkeypatch)
    monkeypatch.setattr(core, "BrokerGateway", _DummyGateway, raising=True)
    monkeypatch.setattr(core._core_impl, "BrokerGateway", _DummyGateway, raising=True)

    logs = []

    cfg = configparser.ConfigParser()
    cfg["KEYS"] = {
        "alpaca_key": "k",
        "alpaca_secret": "s",
        "base_url": "https://paper-api.alpaca.markets",
    }
    cfg["CONFIGURATION"] = {
        "amount_to_trade": "2000",
        "max_daily_loss": "100",
        "agent_mode": "OFF",
        "log_level": "INFO",
    }

    core.TradingEngine(cfg, _DummyDB(), logs.append, lambda *_a, **_k: None)

    assert any("API authentication failed" in line for line in logs)
    assert not any("API Connected but account fetch failed" in line for line in logs)
    assert not any("Sync Failed" in line for line in logs)
