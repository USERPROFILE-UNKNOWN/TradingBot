import configparser
import importlib

from modules.tests.engine_test_stubs import import_engine_core_with_stubs


class _DummyDB:
    def __init__(self, rows=None):
        self._rows = rows or []

    def get_latest_candidates(self, limit=200):
        return list(self._rows)[:limit]


class _DummyEngine:
    def __init__(self, config, db):
        self.config = config
        self.db = db



def _mk_cfg(sections):
    cfg = configparser.ConfigParser()
    for sec, keys in sections.items():
        cfg[sec] = {k: "" for k in keys}
    return cfg


def _svc(monkeypatch, cfg, db):
    import_engine_core_with_stubs(monkeypatch)
    sc = importlib.import_module("modules.engine.scan_coordinator")
    return sc.ScanCoordinatorService(_DummyEngine(cfg, db))


def test_resolver_prefers_active_watchlist_symbols(monkeypatch):
    cfg = _mk_cfg(
        {
            "WATCHLIST_ACTIVE_STOCK": ["AAPL", "MSFT"],
            "WATCHLIST_ARCHIVE_STOCK": ["TSLA"],
        }
    )
    svc = _svc(monkeypatch, cfg, _DummyDB(rows=[{"symbol": "NVDA"}]))

    out = svc._resolve_scan_symbols()

    assert out == ["AAPL", "MSFT"]


def test_resolver_falls_back_to_all_watchlist_then_db_candidates(monkeypatch):
    cfg = _mk_cfg(
        {
            "WATCHLIST_ARCHIVE_STOCK": ["TSLA"],
            "WATCHLIST_ARCHIVE_CRYPTO": ["BTC/USD"],
        }
    )
    svc = _svc(monkeypatch, cfg, _DummyDB(rows=[{"symbol": "NVDA"}, {"symbol": "AMD"}]))

    out = svc._resolve_scan_symbols()

    assert out == ["TSLA", "BTC/USD"]


def test_resolver_uses_db_candidates_when_watchlist_layout_missing(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["CONFIGURATION"] = {"agent_mode": "OFF"}
    svc = _svc(monkeypatch, cfg, _DummyDB(rows=[{"symbol": "NVDA"}, {"symbol": "nvda"}, {"symbol": "AMD"}]))

    out = svc._resolve_scan_symbols()

    assert out == ["NVDA", "AMD"]
