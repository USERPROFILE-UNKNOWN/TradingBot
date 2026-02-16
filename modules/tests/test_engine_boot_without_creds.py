import configparser


from modules.tests.engine_test_stubs import import_engine_core_with_stubs


class _DummyDB:
    def get_active_trades(self):
        return {}


def test_engine_boots_without_credentials_and_does_not_crash(monkeypatch):
    core = import_engine_core_with_stubs(monkeypatch)

    # v5.15.0: connect lives in BrokerGateway (not core.tradeapi).
    def _boom_connect(_self):
        raise RuntimeError("no creds")

    monkeypatch.setattr(core.BrokerGateway, "connect", _boom_connect, raising=True)


    cfg = configparser.ConfigParser()
    cfg["KEYS"] = {"alpaca_key": "", "alpaca_secret": "", "base_url": "https://paper-api.alpaca.markets"}
    cfg["CONFIGURATION"] = {
        "amount_to_trade": "2000",
        "max_daily_loss": "100",
        "agent_mode": "OFF",
        "log_level": "INFO",
    }

    # Should not raise (connect_api failures must be handled internally).
    core.TradingEngine(cfg, _DummyDB(), lambda *_a, **_k: None, lambda *_a, **_k: None)
