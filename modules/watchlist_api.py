"""
Watchlist API (Phase 4 - v5.14.0)

Single source of truth for accessing the watchlist in its new split format.

New watchlist.ini sections:
- WATCHLIST_FAVORITES_STOCK / WATCHLIST_FAVORITES_CRYPTO / WATCHLIST_FAVORITES_ETF
- WATCHLIST_ACTIVE_STOCK    / WATCHLIST_ACTIVE_CRYPTO    / WATCHLIST_ACTIVE_ETF
- WATCHLIST_ARCHIVE_STOCK   / WATCHLIST_ARCHIVE_CRYPTO   / WATCHLIST_ARCHIVE_ETF

Notes:
- Engine/scanners/policy should use ACTIVE by default.
- UI may offer selectors to view FAVORITES / ACTIVE / ARCHIVE / ALL.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


GROUPS = ("FAVORITES", "ACTIVE", "ARCHIVE")
ASSETS = ("STOCK", "CRYPTO", "ETF", "ALL")

SECTION_MAP: Dict[Tuple[str, str], str] = {
    ("FAVORITES", "STOCK"): "WATCHLIST_FAVORITES_STOCK",
    ("FAVORITES", "CRYPTO"): "WATCHLIST_FAVORITES_CRYPTO",
    ("FAVORITES", "ETF"): "WATCHLIST_FAVORITES_ETF",
    ("ACTIVE", "STOCK"): "WATCHLIST_ACTIVE_STOCK",
    ("ACTIVE", "CRYPTO"): "WATCHLIST_ACTIVE_CRYPTO",
    ("ACTIVE", "ETF"): "WATCHLIST_ACTIVE_ETF",
    ("ARCHIVE", "STOCK"): "WATCHLIST_ARCHIVE_STOCK",
    ("ARCHIVE", "CRYPTO"): "WATCHLIST_ARCHIVE_CRYPTO",
    ("ARCHIVE", "ETF"): "WATCHLIST_ARCHIVE_ETF",
}

ALL_SECTIONS: Tuple[str, ...] = tuple(SECTION_MAP.values())


def _is_crypto_symbol(symbol: str) -> bool:
    """
    Heuristic:
    - Treat symbols with a slash (e.g., BTC/USD, SOL/USD) as crypto/pairs.
    - Everything else is treated as stock.
    """
    try:
        s = str(symbol or "").strip().upper()
    except Exception:
        return False
    return ("/" in s)


def _detect_asset(symbol: str, explicit_asset: str = "") -> str:
    a = (explicit_asset or "").strip().upper()
    if a in ("STOCK", "CRYPTO", "ETF"):
        return a
    return "CRYPTO" if _is_crypto_symbol(symbol) else "STOCK"


def _norm_group(group: str) -> str:
    g = (group or "ACTIVE").strip().upper()
    if g == "ALL":
        return "ALL"
    if g not in GROUPS:
        return "ACTIVE"
    return g


def _norm_asset(asset: str) -> str:
    a = (asset or "STOCK").strip().upper()
    if a not in ASSETS:
        return "STOCK"
    return a


def _has_section(cfg: Any, name: str) -> bool:
    try:
        return cfg is not None and hasattr(cfg, "has_section") and cfg.has_section(name)
    except Exception:
        try:
            return cfg is not None and name in cfg
        except Exception:
            return False


def _ensure_section(cfg: Any, name: str) -> None:
    try:
        if hasattr(cfg, "has_section") and hasattr(cfg, "add_section"):
            if not cfg.has_section(name):
                cfg.add_section(name)
            return
    except Exception:
        pass
    # dict-like fallback
    try:
        if isinstance(cfg, dict) and name not in cfg:
            cfg[name] = {}
    except Exception:
        pass


def _section_items(cfg: Any, section: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        if cfg is None:
            return out
        if hasattr(cfg, "__contains__") and section in cfg:
            sec = cfg[section]
            if hasattr(sec, "items"):
                return [(str(k), str(v or "")) for k, v in sec.items()]
        if isinstance(cfg, dict):
            sec = cfg.get(section) or {}
            if hasattr(sec, "items"):
                return [(str(k), str(v or "")) for k, v in sec.items()]
    except Exception:
        return out
    return out


def _read_sections(cfg: Any, sections: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for sec in sections:
        for raw, _sector in _section_items(cfg, sec):
            try:
                sym = str(raw).strip().upper()
            except Exception:
                continue
            if not sym:
                continue
            if sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
    return out


def get_watchlist_entries(cfg: Any, *, group: str = "ACTIVE", asset: str = "ALL") -> List[Dict[str, str]]:
    """Return normalized watchlist rows with optional sector metadata.

    Row shape: {symbol, market, state, sector}
    """
    g = _norm_group(group)
    a = _norm_asset(asset)

    has_new = any(_has_section(cfg, s) for s in ALL_SECTIONS)
    if not has_new:
        return []

    groups = GROUPS if g == "ALL" else (g,)
    assets = ("STOCK", "CRYPTO", "ETF") if a == "ALL" else (a,)

    rows: List[Dict[str, str]] = []
    seen = set()
    for gg in groups:
        for aa in assets:
            sec = SECTION_MAP[(gg, aa)]
            for raw, raw_sector in _section_items(cfg, sec):
                sym = str(raw or "").strip().upper()
                if not sym:
                    continue
                k = (gg, aa, sym)
                if k in seen:
                    continue
                seen.add(k)
                rows.append(
                    {
                        "symbol": sym,
                        "market": aa,
                        "state": gg,
                        "sector": str(raw_sector or "").strip(),
                    }
                )
    return rows


def get_watchlist_symbols(cfg: Any, *, group: str = "ACTIVE", asset: str = "STOCK") -> List[str]:
    """
    Return symbols from the watchlist.

    group:
      - FAVORITES / ACTIVE / ARCHIVE / ALL
    asset:
      - STOCK / CRYPTO / ALL
    """
    g = _norm_group(group)
    a = _norm_asset(asset)

    # Prefer new layout if any new section exists.
    has_new = any(_has_section(cfg, s) for s in ALL_SECTIONS)
    if not has_new:
        return []


    if g == "ALL":
        groups = GROUPS
    else:
        groups = (g,)

    sections: List[str] = []
    for gg in groups:
        if a in ("STOCK", "CRYPTO", "ETF"):
            sections.append(SECTION_MAP[(gg, a)])
        else:
            sections.append(SECTION_MAP[(gg, "STOCK")])
            sections.append(SECTION_MAP[(gg, "CRYPTO")])
            sections.append(SECTION_MAP[(gg, "ETF")])

    return _read_sections(cfg, sections)


def add_watchlist_symbol(
    cfg: Any,
    symbol: str,
    *,
    group: str = "ACTIVE",
    asset: str = "",
    sector: str = "",
) -> bool:
    """Add a single symbol to the appropriate STOCK/CRYPTO/ETF watchlist section."""
    try:
        sym = str(symbol or "").strip().upper()
    except Exception:
        return False
    if not sym:
        return False

    g = _norm_group(group)
    if g == "ALL":
        g = "ACTIVE"

    mkt = _detect_asset(sym, asset)
    sec = SECTION_MAP[(g, mkt)]

    _ensure_section(cfg, sec)
    try:
        cfg[sec][sym] = str(sector or "")
        return True
    except Exception:
        # dict-like fallback
        try:
            if isinstance(cfg, dict):
                d = cfg.setdefault(sec, {})
                d[sym] = str(sector or "")
                return True
        except Exception:
            return False
    return False


def remove_watchlist_symbol(cfg: Any, symbol: str, *, group: str = "ACTIVE", asset: str = "") -> bool:
    """Remove a symbol from STOCK/CRYPTO/ETF section(s) (if present)."""
    try:
        sym = str(symbol or "").strip().upper()
    except Exception:
        return False
    if not sym:
        return False

    g = _norm_group(group)
    if g == "ALL":
        g = "ACTIVE"

    mkt = _detect_asset(sym, asset)
    candidate_secs = [SECTION_MAP[(g, mkt)]]
    if not asset:
        for aa in ("STOCK", "CRYPTO", "ETF"):
            sname = SECTION_MAP[(g, aa)]
            if sname not in candidate_secs:
                candidate_secs.append(sname)

    for sec in candidate_secs:
        try:
            if hasattr(cfg, "__contains__") and sec in cfg and sym in cfg[sec]:
                del cfg[sec][sym]
                return True
        except Exception:
            pass

        try:
            if isinstance(cfg, dict) and sec in cfg:
                d = cfg.get(sec) or {}
                if sym in d:
                    del d[sym]
                    return True
        except Exception:
            pass

    return False


def ensure_watchlist_sections(cfg: Any) -> None:
    """Ensure all new watchlist sections exist (no-ops if cfg is dict-like)."""
    for s in ALL_SECTIONS:
        _ensure_section(cfg, s)
