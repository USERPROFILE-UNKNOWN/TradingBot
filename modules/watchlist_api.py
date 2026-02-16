"""
Watchlist API (Phase 4 - v5.14.0)

Single source of truth for accessing the watchlist in its new split format.

New watchlist.ini sections:
- WATCHLIST_FAVORITES_STOCK / WATCHLIST_FAVORITES_CRYPTO
- WATCHLIST_ACTIVE_STOCK    / WATCHLIST_ACTIVE_CRYPTO
- WATCHLIST_ARCHIVE_STOCK   / WATCHLIST_ARCHIVE_CRYPTO

Notes:
- Engine/scanners/policy should use ACTIVE by default.
- UI may offer selectors to view FAVORITES / ACTIVE / ARCHIVE / ALL.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


GROUPS = ("FAVORITES", "ACTIVE", "ARCHIVE")
ASSETS = ("STOCK", "CRYPTO", "ALL")

SECTION_MAP: Dict[Tuple[str, str], str] = {
    ("FAVORITES", "STOCK"): "WATCHLIST_FAVORITES_STOCK",
    ("FAVORITES", "CRYPTO"): "WATCHLIST_FAVORITES_CRYPTO",
    ("ACTIVE", "STOCK"): "WATCHLIST_ACTIVE_STOCK",
    ("ACTIVE", "CRYPTO"): "WATCHLIST_ACTIVE_CRYPTO",
    ("ARCHIVE", "STOCK"): "WATCHLIST_ARCHIVE_STOCK",
    ("ARCHIVE", "CRYPTO"): "WATCHLIST_ARCHIVE_CRYPTO",
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


def _section_keys(cfg: Any, section: str) -> List[str]:
    try:
        if cfg is None:
            return []
        if hasattr(cfg, "__contains__") and section in cfg:
            sec = cfg[section]
            if hasattr(sec, "keys"):
                return [str(k) for k in sec.keys()]
        if isinstance(cfg, dict):
            sec = cfg.get(section) or {}
            if hasattr(sec, "keys"):
                return [str(k) for k in sec.keys()]
    except Exception:
        return []
    return []


def _read_sections(cfg: Any, sections: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for sec in sections:
        for raw in _section_keys(cfg, sec):
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
        if a in ("STOCK", "CRYPTO"):
            sections.append(SECTION_MAP[(gg, a)])
        else:
            sections.append(SECTION_MAP[(gg, "STOCK")])
            sections.append(SECTION_MAP[(gg, "CRYPTO")])

    return _read_sections(cfg, sections)


def add_watchlist_symbol(cfg: Any, symbol: str, *, group: str = "ACTIVE") -> bool:
    """Add a single symbol to the appropriate STOCK/CRYPTO watchlist section."""
    try:
        sym = str(symbol or "").strip().upper()
    except Exception:
        return False
    if not sym:
        return False

    g = _norm_group(group)
    if g == "ALL":
        g = "ACTIVE"

    asset = "CRYPTO" if _is_crypto_symbol(sym) else "STOCK"
    sec = SECTION_MAP[(g, asset)]

    _ensure_section(cfg, sec)
    try:
        cfg[sec][sym] = ""
        return True
    except Exception:
        # dict-like fallback
        try:
            if isinstance(cfg, dict):
                d = cfg.setdefault(sec, {})
                d[sym] = ""
                return True
        except Exception:
            return False
    return False


def remove_watchlist_symbol(cfg: Any, symbol: str, *, group: str = "ACTIVE") -> bool:
    """Remove a symbol from the appropriate STOCK/CRYPTO section (if present)."""
    try:
        sym = str(symbol or "").strip().upper()
    except Exception:
        return False
    if not sym:
        return False

    g = _norm_group(group)
    if g == "ALL":
        g = "ACTIVE"

    asset = "CRYPTO" if _is_crypto_symbol(sym) else "STOCK"
    sec = SECTION_MAP[(g, asset)]

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
