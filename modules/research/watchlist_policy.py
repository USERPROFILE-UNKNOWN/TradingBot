"""Dynamic watchlist policy (v5.13.1 updateB).

Goals
- Optional auto-update of watchlist based on daily candidates.
- Optional "crypto stable set" selection filtered by liquidity + spread.
- Every applied change is logged and reversible via config backup.

This module is intentionally conservative:
- It only mutates the [WATCHLIST] section.
- It does NOT change live engine logic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _cfg(config: Any, key: str, default: str = "") -> str:
    try:
        return (config.get("CONFIGURATION", key, fallback=default) or default)
    except Exception:
        try:
            sec = config.get("CONFIGURATION", {})
            return (sec.get(key, default) or default)
        except Exception:
            return default


def _cfg_int(config: Any, key: str, default: int) -> int:
    try:
        return int(float(_cfg(config, key, str(default))))
    except Exception:
        return int(default)


def _cfg_float(config: Any, key: str, default: float) -> float:
    try:
        return float(_cfg(config, key, str(default)))
    except Exception:
        return float(default)


def _cfg_bool(config: Any, key: str, default: bool = False) -> bool:
    try:
        v = str(_cfg(config, key, "True" if default else "False")).strip().lower()
        return v in ("1", "true", "yes", "y", "on")
    except Exception:
        return bool(default)


def _is_crypto_symbol(sym: str) -> bool:
    try:
        return "/" in str(sym)
    except Exception:
        return False


def _range_proxy_spread_pct(df: Any) -> Optional[float]:
    """Fallback spread proxy (%).

    If bid/ask is unavailable, we use median((high-low)/close)*100 over the window.
    This is NOT a true spread, but it does correlate with noisy/wide markets.
    """
    try:
        if df is None or getattr(df, "empty", True):
            return None
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        cl = df["close"].astype(float)
        rng = (hi - lo).abs()
        pct = (rng / cl.replace(0.0, float("nan"))) * 100.0
        try:
            pct = pct.replace([float("inf"), float("-inf")], float("nan"))
        except Exception:
            pass
        try:
            v = float(pct.median())
        except Exception:
            v = float(pct.dropna().iloc[-1])
        if v != v:  # NaN
            return None
        return max(0.0, v)
    except Exception:
        return None


def _extract_bid_ask(quote_obj: Any) -> Tuple[Optional[float], Optional[float]]:
    if quote_obj is None:
        return None, None

    # dict-like
    if isinstance(quote_obj, dict):
        bid = None
        ask = None
        for k in ("bid_price", "bid", "bp", "b", "BidPrice", "bidPrice"):
            if k in quote_obj and quote_obj[k] is not None:
                try:
                    bid = float(quote_obj[k])
                    break
                except Exception:
                    pass
        for k in ("ask_price", "ask", "ap", "a", "AskPrice", "askPrice"):
            if k in quote_obj and quote_obj[k] is not None:
                try:
                    ask = float(quote_obj[k])
                    break
                except Exception:
                    pass
        return bid, ask

    # object-like
    bid = None
    ask = None
    for attr in ("bid_price", "bid", "bp", "BidPrice", "bidPrice"):
        try:
            v = getattr(quote_obj, attr)
            if v is not None:
                bid = float(v)
                break
        except Exception:
            pass
    for attr in ("ask_price", "ask", "ap", "AskPrice", "askPrice"):
        try:
            v = getattr(quote_obj, attr)
            if v is not None:
                ask = float(v)
                break
        except Exception:
            pass
    return bid, ask


def _get_quote_spread_pct(api: Any, symbol: str) -> Optional[float]:
    """Best-effort bid/ask spread in percent. Returns None if unavailable."""
    if api is None:
        return None

    symbol = str(symbol).strip().upper()
    if not symbol:
        return None

    candidates = [
        "get_latest_crypto_quotes",
        "get_latest_crypto_quote",
        "get_crypto_latest_quote",
        "get_latest_quote",
        "get_latest_quotes",
    ]

    for fn_name in candidates:
        fn = getattr(api, fn_name, None)
        if not callable(fn):
            continue
        try:
            try:
                res = fn([symbol])
            except Exception:
                res = fn(symbol)
        except Exception:
            continue

        q = None
        try:
            if isinstance(res, dict):
                q = res.get(symbol) or (list(res.values())[0] if res else None)
            elif pd is not None and isinstance(res, pd.DataFrame):
                if not res.empty:
                    q = res.iloc[0].to_dict()
            else:
                # alpaca_trade_api often returns an object with .df
                df = getattr(res, "df", None)
                if pd is not None and df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    q = df.iloc[0].to_dict()
                else:
                    q = res
        except Exception:
            q = res

        bid, ask = _extract_bid_ask(q)
        if bid is None or ask is None:
            continue

        try:
            mid = (bid + ask) / 2.0
            if mid <= 0:
                return None
            return abs(ask - bid) / mid * 100.0
        except Exception:
            return None

    return None


def compute_crypto_stable_set(
    db: Any,
    config: Any,
    *,
    api: Optional[Any] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return a liquidity/spread-filtered set of crypto symbols."""

    lookback = max(50, _cfg_int(config, "crypto_stable_set_lookback_bars", 390))
    max_assets = max(1, _cfg_int(config, "crypto_stable_set_max_assets", 6))
    min_dv = max(0.0, _cfg_float(config, "crypto_stable_set_min_dollar_volume", 5_000_000.0))
    max_spread = max(0.0, _cfg_float(config, "crypto_stable_set_max_spread_pct", 0.5))

    syms: List[str] = []
    try:
        all_syms = db.get_all_symbols() or []
        syms = sorted(list({str(s).upper().strip() for s in all_syms if _is_crypto_symbol(str(s))}))
    except Exception:
        syms = []

    kept: List[Tuple[str, float, Optional[float], str]] = []
    skipped = 0

    for sym in syms:
        try:
            df = db.get_history(sym, lookback)
        except Exception:
            skipped += 1
            continue

        if df is None or getattr(df, "empty", True):
            skipped += 1
            continue

        dv = 0.0
        try:
            dv = float((df["close"].astype(float) * df["volume"].astype(float)).sum())
        except Exception:
            try:
                dv = float(df["volume"].astype(float).sum())
            except Exception:
                dv = 0.0

        if dv < float(min_dv):
            continue

        spread = _get_quote_spread_pct(api, sym)
        spread_src = "quote"
        if spread is None:
            spread = _range_proxy_spread_pct(df)
            spread_src = "range_proxy"

        if spread is None:
            continue

        if float(spread) > float(max_spread):
            continue

        kept.append((sym, float(dv), float(spread), spread_src))

    kept.sort(key=lambda x: x[1], reverse=True)
    out = [k[0] for k in kept[:max_assets]]

    meta = {
        "lookback_bars": lookback,
        "max_assets": max_assets,
        "min_dollar_volume": min_dv,
        "max_spread_pct": max_spread,
        "universe_size": len(syms),
        "skipped": skipped,
        "selected": [
            {"symbol": s, "dollar_volume": dv, "spread_pct": sp, "spread_source": src}
            for (s, dv, sp, src) in kept[:max_assets]
        ],
    }

    if callable(log):
        try:
            log(f"[WATCHLIST] Crypto stable set selected: {out} (universe={len(syms)}, lookback={lookback})")
        except Exception:
            pass

    return out, meta


def apply_watchlist_policy(
    config: Any,
    db: Any,
    paths: Dict[str, str],
    *,
    api: Optional[Any] = None,
    log: Optional[Callable[[str], None]] = None,
    source: str = "watchlist_policy",
    apply_mode: Optional[str] = None,
    backup_cb: Optional[Callable[[], Optional[str]]] = None,
    write_cb: Optional[Callable[[], None]] = None,
    refresh_cb: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Apply the configured watchlist policy.

    Returns a dict with keys: changed(bool), added(list), removed(list), new_watchlist(list), batch_id(str), backup_dir(str|None).
    """

    ts = datetime.now(timezone.utc)
    batch_id = ts.strftime("%Y.%m.%d_%H.%M.%S")

    # Config knobs (all default OFF)
    enabled = _cfg_bool(config, "watchlist_auto_update_enabled", False)
    if not enabled and apply_mode is None:
        return {
            "changed": False,
            "reason": "watchlist_auto_update_enabled is OFF",
            "batch_id": batch_id,
        }

    mode = (apply_mode or _cfg(config, "watchlist_auto_update_mode", "ADD") or "ADD").strip().upper()
    max_add = max(0, _cfg_int(config, "watchlist_auto_update_max_add", 5))
    max_total = max(1, _cfg_int(config, "watchlist_auto_update_max_total", 20))
    min_score = _cfg_float(config, "watchlist_auto_update_min_score", 0.0)

    crypto_enabled = _cfg_bool(config, "crypto_stable_set_enabled", False)
    crypto_replace = _cfg_bool(config, "crypto_stable_set_replace_existing", True)

    # Existing watchlist (Phase 4 v5.14.0: ACTIVE universe)
    existing_list: List[str] = []
    try:
        from ..watchlist_api import get_watchlist_symbols, ensure_watchlist_sections
        ensure_watchlist_sections(config)
        existing_list = [str(s).strip().upper() for s in get_watchlist_symbols(config, group="ACTIVE", asset="ALL") if str(s).strip()]
    except Exception:
        existing_list = []

    # Preserve order, de-dup + keep simple rejection reasons for policy transparency.
    reject_reasons: Dict[str, List[str]] = {}

    def _reject(sym: str, reason: str) -> None:
        ss = str(sym or "").strip().upper()
        if not ss:
            return
        arr = reject_reasons.setdefault(ss, [])
        if reason not in arr:
            arr.append(reason)

    # Candidate-driven selection (for non-crypto)
    candidate_syms: List[str] = []
    try:
        today = ts.strftime("%Y-%m-%d")
        df = db.get_latest_candidates(scan_date=today, limit=500)
        if df is not None and not getattr(df, "empty", True):
            try:
                # filter by score
                for _i, r in df.iterrows():
                    sym = str(r.get("symbol", "")).strip().upper()
                    score = float(r.get("score", 0.0) or 0.0)
                    if not sym:
                        continue
                    if score < float(min_score):
                        _reject(sym, "score_below_min")
                        continue
                    if _is_crypto_symbol(sym):
                        _reject(sym, "crypto_managed_separately")
                        continue
                    candidate_syms.append(sym)
            except Exception:
                pass
    except Exception:
        pass

    seen = set()
    cand_unique: List[str] = []
    for s in candidate_syms:
        if s in seen:
            _reject(s, "duplicate_candidate")
            continue
        seen.add(s)
        cand_unique.append(s)

    # Build new list
    new_non_crypto: List[str] = []
    if mode == "REPLACE":
        new_non_crypto = []
    else:
        # keep existing non-crypto first
        new_non_crypto = [s for s in existing_list if not _is_crypto_symbol(s)]

    if cand_unique:
        if mode == "REPLACE":
            new_non_crypto = cand_unique[:max_total]
        else:
            added_count = 0
            existing_set = set(new_non_crypto)
            for s in cand_unique:
                if s in existing_set:
                    _reject(s, "already_active")
                    continue
                if max_add > 0 and added_count >= max_add:
                    _reject(s, "max_add_limit")
                    continue
                new_non_crypto.append(s)
                existing_set.add(s)
                added_count += 1

    # Crypto stable set
    crypto_meta: Dict[str, Any] = {}
    crypto_syms: List[str] = []
    if crypto_enabled:
        crypto_syms, crypto_meta = compute_crypto_stable_set(db, config, api=api, log=log)

    # Existing crypto symbols (only used if we are NOT replacing)
    existing_crypto = [s for s in existing_list if _is_crypto_symbol(s)]
    if crypto_enabled:
        if not crypto_replace:
            # keep existing cryptos + ensure stable set present
            merged = []
            seen2 = set()
            for s in existing_crypto + crypto_syms:
                if s in seen2:
                    continue
                seen2.add(s)
                merged.append(s)
            crypto_syms = merged
        # else: replace => crypto_syms already defines crypto portion
    else:
        # no crypto policy => keep existing crypto entries
        crypto_syms = existing_crypto

    # Enforce max_total while keeping crypto stable set priority
    if max_total > 0:
        if crypto_enabled and crypto_syms:
            room = max_total - len(crypto_syms)
            if room < 0:
                crypto_syms = crypto_syms[:max_total]
                new_non_crypto = []
            else:
                if len(new_non_crypto) > room:
                    for d in new_non_crypto[room:]:
                        _reject(d, "max_total_limit")
                new_non_crypto = new_non_crypto[:room]
        else:
            # truncate combined
            pass

    new_list = new_non_crypto + crypto_syms

    # If max_total still exceeded (non-crypto only case)
    if max_total > 0 and len(new_list) > max_total:
        for d in new_list[max_total:]:
            _reject(d, "max_total_limit")
        new_list = new_list[:max_total]

    # Compute deltas
    old_set = set(existing_list)
    new_set = set(new_list)
    added = sorted(list(new_set - old_set))
    removed = sorted(list(old_set - new_set))

    if not added and not removed:
        return {
            "changed": False,
            "reason": "no changes",
            "batch_id": batch_id,
            "new_watchlist": new_list,
            "rejected": reject_reasons,
        }

    # Backup first
    backup_dir: Optional[str] = None
    if callable(backup_cb):
        try:
            backup_dir = backup_cb()
        except Exception:
            backup_dir = None

    # Mutate config (Phase 4 v5.14.0): rewrite ACTIVE sections; archive removals.
    try:
        from ..watchlist_api import ensure_watchlist_sections
        ensure_watchlist_sections(config)
    except Exception:
        pass

    try:
        # Clear ACTIVE sections
        for sec in ("WATCHLIST_ACTIVE_STOCK", "WATCHLIST_ACTIVE_CRYPTO"):
            try:
                if not config.has_section(sec):
                    config.add_section(sec)
                for k in list(config[sec].keys()):
                    try:
                        config.remove_option(sec, k)
                    except Exception:
                        try:
                            del config[sec][k]
                        except Exception:
                            pass
            except Exception:
                pass

        # Apply new ACTIVE list
        for s in new_list:
            try:
                sec = "WATCHLIST_ACTIVE_CRYPTO" if _is_crypto_symbol(s) else "WATCHLIST_ACTIVE_STOCK"
                config[sec][s] = ""
            except Exception:
                pass

        # Archive removals so UI can view them
        for s in removed:
            try:
                sec = "WATCHLIST_ARCHIVE_CRYPTO" if _is_crypto_symbol(s) else "WATCHLIST_ARCHIVE_STOCK"
                if not config.has_section(sec):
                    config.add_section(sec)
                config[sec][s] = ""
            except Exception:
                pass
    except Exception:
        return {
            "changed": False,
            "reason": "failed to mutate watchlist",
            "batch_id": batch_id,
        }

    # Persist
    if callable(write_cb):
        try:
            write_cb()
        except Exception:
            pass

    if callable(refresh_cb):
        try:
            refresh_cb()
        except Exception:
            pass

    # Audit log file
    try:
        logs_root = paths.get("logs") or ""
        out_dir = os.path.join(logs_root, "research") if logs_root else ""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"watchlist_policy_{batch_id}.json")
            payload = {
                "timestamp": ts.isoformat(),
                "batch_id": batch_id,
                "source": source,
                "mode": mode,
                "min_score": min_score,
                "max_add": max_add,
                "max_total": max_total,
                "crypto_stable_set_enabled": crypto_enabled,
                "crypto_meta": crypto_meta,
                "backup_dir": backup_dir,
                "before": existing_list,
                "after": new_list,
                "added": added,
                "removed": removed,
                "rejected": reject_reasons,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Audit into DB (best-effort)
    try:
        if hasattr(db, "log_watchlist_audit") and callable(getattr(db, "log_watchlist_audit")):
            db.log_watchlist_audit(
                batch_id=batch_id,
                source=source,
                mode=mode,
                before=existing_list,
                after=new_list,
                added=added,
                removed=removed,
                meta={"backup_dir": backup_dir, "crypto_meta": crypto_meta},
            )
    except Exception:
        pass

    if callable(log):
        try:
            log(f"[WATCHLIST] ✅ Policy applied ({mode}). Added={added} Removed={removed}")
        except Exception:
            pass

    return {
        "changed": True,
        "batch_id": batch_id,
        "backup_dir": backup_dir,
        "new_watchlist": new_list,
        "added": added,
        "removed": removed,
        "rejected": reject_reasons,
        "crypto_meta": crypto_meta,
    }
