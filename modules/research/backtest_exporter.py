"""Backtest export utilities.

v5.12.8 updateA_backtest_exports

Exports a stable bundle to logs/backtest/.

Important:
- Does NOT export secrets (skips the KEYS section).
- Trade-by-trade metrics are not available yet because the current backtest
  pipeline stores only aggregate per-symbol results. This exporter records that
  limitation explicitly in the output schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_get_section(config: Any, section: str) -> Dict[str, str]:
    try:
        if config is None:
            return {}
        if hasattr(config, "has_section") and config.has_section(section):
            return {k: str(v) for k, v in config[section].items()}
    except Exception:
        pass
    return {}


def _extract_strategy_sections(config: Any) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    try:
        if config is None or not hasattr(config, "sections"):
            return out
        for sec in config.sections():
            if str(sec).startswith("STRATEGY_"):
                try:
                    out[str(sec)] = {k: str(v) for k, v in config[sec].items()}
                except Exception:
                    out[str(sec)] = {}
    except Exception:
        return out
    return out


def _params_hash(payload: Dict[str, Any]) -> str:
    try:
        b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        b = repr(payload).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def export_backtest_bundle(
    *,
    backtest_df: Any,
    config: Any,
    out_dir: str,
    app_version: str = "",
    app_release: str = "",
    include_csv: bool = True,
) -> Optional[Dict[str, str]]:
    """Export a backtest bundle.

    Returns dict with keys: json, csv (optional).
    """
    try:
        if backtest_df is None:
            return None
        # pandas DataFrame contract
        if getattr(backtest_df, "empty", False):
            return None
    except Exception:
        return None

    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    base = f"backtest_bundle_{ts}"
    json_path = os.path.join(out_dir, f"{base}.json")
    csv_path = os.path.join(out_dir, f"{base}_results.csv")

    # ---- Snapshot rows (stable) ----
    try:
        cols = list(getattr(backtest_df, "columns", []))
        # preserve column order, but coerce into python primitives for JSON
        rows = []
        for _, r in backtest_df.iterrows():
            d = {}
            for c in cols:
                v = r[c]
                try:
                    # pandas/numpy scalars
                    if hasattr(v, "item"):
                        v = v.item()
                except Exception:
                    pass
                d[str(c)] = v
            rows.append(d)
    except Exception:
        return None

    # ---- Params (hashable) ----
    cfg_backtest = {
        k: v
        for k, v in _safe_get_section(config, "CONFIGURATION").items()
        if str(k).lower().startswith("backtest_")
    }
    strategies = _extract_strategy_sections(config)

    params_payload = {
        "backtest": cfg_backtest,
        "strategies": strategies,
    }
    params_hash = _params_hash(params_payload)

    # ---- Aggregates ----
    symbols = []
    best_counts: Dict[str, int] = {}
    by_strategy: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        sym = row.get("symbol")
        if sym is not None:
            symbols.append(sym)
        bs = row.get("best_strategy")
        if bs is None:
            bs = "None"
        bs = str(bs)
        best_counts[bs] = best_counts.get(bs, 0) + 1

    # Strategy-level aggregates based on existing columns (PL_* and Trades_*)
    try:
        pl_cols = [c for c in cols if str(c).startswith("PL_")]
        tr_cols = [c for c in cols if str(c).startswith("Trades_")]

        # discover strategy names from PL_ prefix
        strat_names = sorted({str(c)[3:] for c in pl_cols})
        for s in strat_names:
            pl_key = f"PL_{s}"
            tr_key = f"Trades_{s}"

            pl_sum = 0.0
            trades_sum = 0
            n = 0
            n_pos = 0

            for row in rows:
                try:
                    v = float(row.get(pl_key, 0.0) or 0.0)
                except Exception:
                    v = 0.0
                try:
                    t = int(float(row.get(tr_key, 0) or 0))
                except Exception:
                    t = 0

                pl_sum += v
                trades_sum += t
                n += 1
                if v > 0:
                    n_pos += 1

            by_strategy[s] = {
                "symbols": n,
                "symbols_positive": n_pos,
                "symbol_win_rate": (n_pos / n) if n else None,
                "profit_sum": round(pl_sum, 4),
                "profit_mean": round(pl_sum / n, 6) if n else None,
                "trades_sum": trades_sum,
                "trades_mean": round(trades_sum / n, 6) if n else None,
            }
    except Exception:
        by_strategy = {}

    # ---- High-level metrics (best_strategy snapshot) ----
    best_profit_sum = 0.0
    best_pos = 0
    n_rows = len(rows)

    for row in rows:
        try:
            bp = float(row.get("best_profit", 0.0) or 0.0)
        except Exception:
            bp = 0.0
        best_profit_sum += bp
        if bp > 0:
            best_pos += 1

    metrics = {
        "pnl_total_best_profit": round(best_profit_sum, 4),
        "symbol_win_rate_best_profit": (best_pos / n_rows) if n_rows else None,
        # placeholders (trade-level stats not available yet)
        "max_drawdown": None,
        "profit_factor": None,
        "avg_R": None,
        "turnover": None,
        "slippage_model": cfg_backtest.get("backtest_slippage_model"),
        "notes": {
            "trade_level_metrics": "Not available: current backtest pipeline stores only aggregate per-symbol results."
        },
    }

    bundle = {
        "schema": {"name": "TradingBot.backtest_bundle", "version": 1},
        "exported_at_utc": _utc_iso_now(),
        "app": {"version": app_version, "release": app_release},
        "params": params_payload,
        "params_hash": params_hash,
        "universe": {"symbols": symbols, "count": len(symbols)},
        "results": {"rows": rows, "row_count": len(rows)},
        "aggregates": {"best_strategy_counts": best_counts, "by_strategy": by_strategy},
        "metrics": metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    out = {"json": json_path}

    if include_csv:
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k) for k in cols})
            out["csv"] = csv_path
        except Exception:
            # csv is optional
            pass

    return out
