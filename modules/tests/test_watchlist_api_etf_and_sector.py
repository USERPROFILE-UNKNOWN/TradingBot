import configparser

from modules.watchlist_api import (
    add_watchlist_symbol,
    ensure_watchlist_sections,
    get_watchlist_entries,
    get_watchlist_symbols,
    remove_watchlist_symbol,
)


def _cfg():
    c = configparser.ConfigParser()
    c.optionxform = str
    ensure_watchlist_sections(c)
    return c


def test_etf_symbols_supported_and_listed_in_all():
    c = _cfg()
    assert add_watchlist_symbol(c, "SOXL", group="ACTIVE", asset="ETF", sector="Sector")
    syms = get_watchlist_symbols(c, group="ACTIVE", asset="ALL")
    assert "SOXL" in syms
    assert "SOXL" in c["WATCHLIST_ACTIVE_ETF"]


def test_entries_include_sector_market_state():
    c = _cfg()
    add_watchlist_symbol(c, "AMD", group="ACTIVE", asset="STOCK", sector="Electronic technology")
    add_watchlist_symbol(c, "BTC/USD", group="ACTIVE", asset="CRYPTO", sector="Cryptocurrencies")
    add_watchlist_symbol(c, "TQQQ", group="ARCHIVE", asset="ETF", sector="Size and style")

    rows = get_watchlist_entries(c, group="ALL", asset="ALL")
    idx = {(r["symbol"], r["state"]): r for r in rows}

    assert idx[("AMD", "ACTIVE")]["market"] == "STOCK"
    assert idx[("AMD", "ACTIVE")]["sector"] == "Electronic technology"
    assert idx[("BTC/USD", "ACTIVE")]["market"] == "CRYPTO"
    assert idx[("TQQQ", "ARCHIVE")]["market"] == "ETF"


def test_remove_symbol_from_etf_section():
    c = _cfg()
    add_watchlist_symbol(c, "SQQQ", group="ACTIVE", asset="ETF")
    assert remove_watchlist_symbol(c, "SQQQ", group="ACTIVE", asset="ETF")
    assert "SQQQ" not in c["WATCHLIST_ACTIVE_ETF"]
