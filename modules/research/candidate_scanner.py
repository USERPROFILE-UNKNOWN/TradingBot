"""Candidate scanner (v5.13.1 Update A).

Goal
- Produce a daily ranked list of symbols to consider for research/backtesting.
- Store scan results into the DB (candidates table) for UI consumption.

Notes
- Provider/API preferred, but this implementation is intentionally conservative:
  it uses already-stored historical_prices data by default, and only uses the
  live API handle (engine.api) as an optional future extension.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass
class CandidateRow:
    symbol: str
    score: float
    ret_lookback: float
    dollar_volume: float
    volatility: float
    bars: int
    universe: str
    details: Dict[str, Any]


class CandidateScanner:
    def __init__(self, db: Any, config: Any, log: Optional[Any] = None):
        self.db = db
        self.config = config
        self.log = log

    # -----------------------------
    # Public
    # -----------------------------
    def scan_today(self, *, api: Optional[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Run a scan and persist results. Returns (scan_id, rows)."""
        scan_ts = datetime.now(timezone.utc)
        scan_id = scan_ts.strftime("%Y.%m.%d_%H.%M.%S")
        scan_date = scan_ts.strftime("%Y-%m-%d")

        enabled = self._cfg_bool("candidate_scanner_enabled", False)
        if not enabled:
            # still return last cached result; UI can decide
            return scan_id, []

        universe_mode = (self._cfg("candidate_scanner_universe_mode", "WATCHLIST") or "WATCHLIST").strip().upper()
        limit = self._cfg_int("candidate_scanner_limit", 20)
        lookback_bars = self._cfg_int("candidate_scanner_lookback_bars", 390)
        min_bars = self._cfg_int("candidate_scanner_min_bars", 120)
        min_dv = float(self._cfg("candidate_scanner_min_dollar_volume", "1000000") or 1000000)
        include_neg = self._cfg_bool("candidate_scanner_include_negative_movers", True)

        symbols = self._get_universe(universe_mode)
        if not symbols:
            return scan_id, []

        rows: List[CandidateRow] = []
        for sym in symbols:
            r = self._score_symbol(
                sym,
                lookback_bars=lookback_bars,
                min_bars=min_bars,
                min_dollar_volume=min_dv,
                include_negative_movers=include_neg,
                universe=universe_mode,
            )
            if r is not None:
                rows.append(r)

        rows.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        rows = rows[: max(1, int(limit))]

        out: List[Dict[str, Any]] = []
        for r in rows:
            details_json = json.dumps(r.details, ensure_ascii=False, separators=(",", ":"))
            out.append({
                "scan_ts": scan_ts.isoformat(),
                "scan_date": scan_date,
                "symbol": r.symbol,
                "score": float(r.score),
                "ret_lookback": float(r.ret_lookback),
                "dollar_volume": float(r.dollar_volume),
                "volatility": float(r.volatility),
                "bars": int(r.bars),
                "universe": r.universe,
                "details_json": details_json,
            })

        try:
            if out:
                self.db.save_candidates(scan_id, out)
        except Exception:
            pass

        return scan_id, out


    def score_single_symbol(
        self,
        symbol: str,
        *,
        universe: str = "SCANNER",
        extra_details: Optional[Dict[str, Any]] = None,
        force_accept: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Score one symbol using the same scoring model as scan_today.

        Intended for real-time candidate sources (e.g., external signal feeds).
        Returns a DB-ready candidate row dict, or None if symbol is empty.

        Notes:
        - Uses config's lookback bars for consistent scoring.
        - For real-time sources we relax min-bars / min-dollar-volume gating,
          but still falls back safely if history is missing.
        """
        sym = str(symbol or "").strip().upper()
        if not sym:
            return None

        scan_ts = datetime.now(timezone.utc)
        scan_date = scan_ts.strftime("%Y-%m-%d")

        lookback_bars = self._cfg_int("candidate_scanner_lookback_bars", 390)
        # Relax gating for real-time sources; we still rely on history if available.
        min_bars = 1
        min_dv = 0.0

        r = None
        try:
            r = self._score_symbol(
                sym,
                lookback_bars=lookback_bars,
                min_bars=min_bars,
                min_dollar_volume=min_dv,
                include_negative_movers=True,
                universe=(universe or "SCANNER").strip().upper(),
            )
        except Exception:
            r = None

        details: Dict[str, Any] = {}
        if r is None:
            if not force_accept:
                return None
            details = {
                "reason": "no_history_or_unscoreable",
                "last_price": None,
                "bars": 0,
                "ret_pct": 0.0,
                "dollar_volume": 0.0,
                "volatility": 0.0,
            }
            score = 0.0
            ret_lookback = 0.0
            dv = 0.0
            vol = 0.0
            bars = 0
            uni = (universe or "SCANNER").strip().upper()
        else:
            details = dict(r.details or {})
            score = float(r.score or 0.0)
            ret_lookback = float(r.ret_lookback or 0.0)
            dv = float(r.dollar_volume or 0.0)
            vol = float(r.volatility or 0.0)
            bars = int(r.bars or 0)
            uni = (r.universe or universe or "SCANNER").strip().upper()

        if extra_details:
            try:
                details.update({k: v for k, v in dict(extra_details).items() if k})
            except Exception:
                pass

        details_json = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
        return {
            "scan_ts": scan_ts.isoformat(),
            "scan_date": scan_date,
            "symbol": sym,
            "score": float(score),
            "ret_lookback": float(ret_lookback),
            "dollar_volume": float(dv),
            "volatility": float(vol),
            "bars": int(bars),
            "universe": uni,
            "details_json": details_json,
        }

    # -----------------------------
    # Internals
    # -----------------------------
    def _get_universe(self, mode: str) -> List[str]:
        mode = (mode or "WATCHLIST").strip().upper()
        # Config list
        if mode == "CONFIG_LIST":
            raw = self._cfg("candidate_scanner_universe", "") or ""
            syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
            return sorted(list(dict.fromkeys(syms)))

        if mode == "ALL_DB_SYMBOLS":
            try:
                syms = self.db.get_all_symbols() or []
                return sorted(list(dict.fromkeys([str(s).strip().upper() for s in syms if str(s).strip()])))
            except Exception:
                return []

        # Default: WATCHLIST (Phase 4 v5.14.0: ACTIVE universe)
        try:
            from ..watchlist_api import get_watchlist_symbols
            syms = [str(s).strip().upper() for s in get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL")]
            return sorted(list(dict.fromkeys([s for s in syms if s])))
        except Exception:
            return []

    def _score_symbol(
        self,
        symbol: str,
        *,
        lookback_bars: int,
        min_bars: int,
        min_dollar_volume: float,
        include_negative_movers: bool,
        universe: str,
    ) -> Optional[CandidateRow]:
        try:
            df = self.db.get_history(symbol, int(max(10, lookback_bars)))
        except Exception:
            return None

        if df is None or getattr(df, "empty", True):
            return None

        bars = int(len(df))
        if bars < int(min_bars):
            return None

        try:
            first_open = float(df["open"].iloc[0])
            last_close = float(df["close"].iloc[-1])
        except Exception:
            return None

        if first_open <= 0 or last_close <= 0:
            return None

        ret = (last_close - first_open) / first_open  # fraction
        if (not include_negative_movers) and ret < 0:
            return None

        # Dollar volume approximation over the window
        dv = 0.0
        try:
            dv = float((df["close"] * df["volume"]).sum())
        except Exception:
            try:
                dv = float(df["volume"].sum())
            except Exception:
                dv = 0.0

        if dv < float(min_dollar_volume):
            return None

        # Volatility: std of minute returns (%)
        vol = 0.0
        try:
            if np is not None:
                rets = df["close"].pct_change().dropna().to_numpy()
                if rets.size > 5:
                    vol = float(np.std(rets) * 100.0)
        except Exception:
            vol = 0.0

        ret_pct = float(ret * 100.0)

        # Simple composite score
        # - emphasize movers
        # - reward liquidity (log-scaled dollar volume)
        # - mild reward for volatility
        try:
            dv_term = math.log10(max(dv, 1.0))
        except Exception:
            dv_term = 0.0

        score = abs(ret_pct) * 1.0 + dv_term * 2.0 + vol * 0.5

        details = {
            "ret_pct": ret_pct,
            "last_price": last_close,
            "first_open": first_open,
            "dollar_volume": dv,
            "volatility": vol,
            "bars": bars,
        }

        return CandidateRow(
            symbol=str(symbol).upper(),
            score=float(score),
            ret_lookback=float(ret_pct),
            dollar_volume=float(dv),
            volatility=float(vol),
            bars=bars,
            universe=universe,
            details=details,
        )

    def _cfg(self, key: str, default: str = "") -> str:
        try:
            return (self.config.get("CONFIGURATION", key, fallback=default) or default)
        except Exception:
            return default

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(float(self._cfg(key, str(default))))
        except Exception:
            return int(default)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        try:
            v = str(self._cfg(key, "True" if default else "False")).strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        except Exception:
            return bool(default)
