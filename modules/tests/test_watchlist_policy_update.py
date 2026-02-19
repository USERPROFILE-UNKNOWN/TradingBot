import configparser

from modules.research.watchlist_policy import apply_watchlist_policy


class _DF:
    def __init__(self, rows):
        self._rows = rows
        self.empty = not bool(rows)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


class _DB:
    def __init__(self, rows):
        self._rows = rows
        self.audits = []

    def get_latest_candidates(self, scan_date=None, limit=500):
        return _DF(self._rows[:limit])

    def get_all_symbols(self):
        return []

    def log_watchlist_audit(self, **kwargs):
        self.audits.append(kwargs)


def _cfg():
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {
        "watchlist_auto_update_enabled": "True",
        "watchlist_auto_update_mode": "ADD",
        "watchlist_auto_update_max_add": "1",
        "watchlist_auto_update_max_total": "5",
        "watchlist_auto_update_min_score": "10",
        "crypto_stable_set_enabled": "False",
    }
    c["WATCHLIST_ACTIVE_STOCK"] = {"AAPL": ""}
    c["WATCHLIST_ACTIVE_CRYPTO"] = {}
    c["WATCHLIST_ARCHIVE_STOCK"] = {}
    c["WATCHLIST_ARCHIVE_CRYPTO"] = {}
    return c


def test_watchlist_policy_returns_rejection_reasons(tmp_path):
    db = _DB([
        {"symbol": "AAPL", "score": 99},
        {"symbol": "MSFT", "score": 20},
        {"symbol": "NVDA", "score": 18},
        {"symbol": "TSLA", "score": 5},
        {"symbol": "BTC/USD", "score": 50},
    ])
    cfg = _cfg()
    paths = {"logs": str(tmp_path)}

    res = apply_watchlist_policy(cfg, db, paths)

    assert res.get("changed") is True
    rejected = res.get("rejected") or {}
    assert "AAPL" in rejected and "ALREADY_ACTIVE" in rejected["AAPL"]
    assert "NVDA" in rejected and "MAX_ADD" in rejected["NVDA"]
    assert "TSLA" in rejected and "LOW_SCORE" in rejected["TSLA"]
    assert "BTC/USD" in rejected and "CRYPTO_MANAGED_SEPARATELY" in rejected["BTC/USD"]


class _DBCryptoEmpty(_DB):
    def get_all_symbols(self):
        return ["BTC/USD", "ETH/USD"]

    def get_history(self, _sym, _limit):
        return None


def test_watchlist_policy_empty_crypto_stable_set_is_noop(tmp_path):
    db = _DBCryptoEmpty([])
    cfg = _cfg()
    cfg["CONFIGURATION"]["crypto_stable_set_enabled"] = "True"
    cfg["CONFIGURATION"]["crypto_stable_set_replace_existing"] = "True"
    cfg["WATCHLIST_ACTIVE_CRYPTO"] = {"BTC/USD": "", "ETH/USD": ""}
    paths = {"logs": str(tmp_path)}

    res = apply_watchlist_policy(cfg, db, paths)

    assert res.get("changed") is False
    assert set(res.get("new_watchlist") or []) == {"AAPL", "BTC/USD", "ETH/USD"}
    rejected = res.get("rejected") or {}
    assert "BTC/USD" in rejected and "INSUFFICIENT_VOLUME_DATA_NOOP" in rejected["BTC/USD"]
    assert "ETH/USD" in rejected and "INSUFFICIENT_VOLUME_DATA_NOOP" in rejected["ETH/USD"]
