import configparser
import importlib.util
import pathlib
import sys
import types


class _DummyLogger:
    def warning(self, *_a, **_k):
        pass

    def error(self, *_a, **_k):
        pass


def _load_gateway_module(monkeypatch):
    fake_tradeapi = types.SimpleNamespace(REST=None, rest=types.SimpleNamespace(APIError=Exception))
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", fake_tradeapi)
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(exceptions=types.SimpleNamespace(RequestException=Exception)))

    path = pathlib.Path(__file__).resolve().parents[1] / "engine" / "broker_gateway.py"
    spec = importlib.util.spec_from_file_location("test_broker_gateway_module", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_broker_gateway_falls_back_to_scoped_credentials_when_active_keys_missing(monkeypatch):
    captured = {}

    class _FakeREST:
        def __init__(self, key, secret, base_url, api_version=None):
            captured["key"] = key
            captured["secret"] = secret
            captured["base_url"] = base_url
            captured["api_version"] = api_version

    bg = _load_gateway_module(monkeypatch)
    monkeypatch.setattr(bg.tradeapi, "REST", _FakeREST, raising=True)

    cfg = configparser.ConfigParser()
    cfg["KEYS"] = {
        "paper_trading": "False",
        "alpaca_key": "",
        "alpaca_secret": "",
        "paper_alpaca_key": "paper_k",
        "paper_alpaca_secret": "paper_s",
        "paper_base_url": "https://paper-api.alpaca.markets",
        "live_alpaca_key": "live_k",
        "live_alpaca_secret": "live_s",
        "live_base_url": "https://api.alpaca.markets",
    }

    gw = bg.BrokerGateway(cfg, _DummyLogger())
    gw.connect()

    assert captured["key"] == "live_k"
    assert captured["secret"] == "live_s"
    assert captured["base_url"] == "https://api.alpaca.markets"


def test_broker_gateway_can_switch_environment(monkeypatch):
    calls = []

    class _FakeREST:
        def __init__(self, key, secret, base_url, api_version=None):
            calls.append((key, secret, base_url, api_version))

    bg = _load_gateway_module(monkeypatch)
    monkeypatch.setattr(bg.tradeapi, "REST", _FakeREST, raising=True)

    cfg = configparser.ConfigParser()
    cfg["KEYS"] = {
        "paper_trading": "True",
        "alpaca_key": "paper_k",
        "alpaca_secret": "paper_s",
        "paper_alpaca_key": "paper_k",
        "paper_alpaca_secret": "paper_s",
        "paper_base_url": "https://paper-api.alpaca.markets",
        "live_alpaca_key": "live_k",
        "live_alpaca_secret": "live_s",
        "live_base_url": "https://api.alpaca.markets",
    }

    gw = bg.BrokerGateway(cfg, _DummyLogger())
    gw.connect()
    assert gw.try_switch_environment() is True

    assert calls[0][2] == "https://paper-api.alpaca.markets"
    assert calls[1][2] == "https://api.alpaca.markets"
