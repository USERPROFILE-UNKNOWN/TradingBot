"""Backtest runner utilities.

v5.13.0 updateB_architect_backtest_orchestration

Purpose
- Evaluate a queue of "Architect" variants (genomes) across a symbol universe.
- Return a DataFrame that can be shown in Backtest Lab without rebuilding DB tables.

Constraints
- No auto-apply to live logic.
- Best-effort safety: swallow per-symbol failures but continue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from .architect import TheArchitect

ProgressFn = Callable[[str, float], None]


@dataclass(frozen=True)
class Variant:
    vid: str
    source_symbol: str
    genome: Dict[str, Any]
    notes: Dict[str, Any]


def _safe_id(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "VAR"
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "VAR"


def _coerce_genome(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a genome dict into the keys expected by TheArchitect._simulate()."""
    g: Dict[str, Any] = {}

    # Expected keys: rsi, sl, tp, ema
    try:
        g["rsi"] = int(float(d.get("rsi", d.get("RSI", d.get("rsi_buy", 30))) or 30))
    except Exception:
        g["rsi"] = 30

    try:
        g["sl"] = float(d.get("sl", d.get("SL", 0.03)) or 0.03)
    except Exception:
        g["sl"] = 0.03

    try:
        g["tp"] = float(d.get("tp", d.get("TP", 2.0)) or 2.0)
    except Exception:
        g["tp"] = 2.0

    try:
        g["ema"] = bool(d.get("ema", d.get("EMA", False)))
    except Exception:
        g["ema"] = False

    return g


def normalize_variants(queue_items: List[Dict[str, Any]]) -> List[Variant]:
    """Convert TradingApp queue items into a stable Variant list."""
    out: List[Variant] = []
    seen = set()

    for item in (queue_items or []):
        try:
            vid = _safe_id(item.get("id") or item.get("vid") or "")
            src = str(item.get("source_symbol") or item.get("symbol") or "").strip()
            raw = item.get("genome") or item
            genome = _coerce_genome(dict(raw))

            key = (genome.get("rsi"), genome.get("sl"), genome.get("tp"), bool(genome.get("ema")))
            if key in seen:
                continue
            seen.add(key)

            notes = {}
            for k in ("profit", "trades", "win_rate", "score"):
                if k in item:
                    notes[k] = item.get(k)

            if not vid:
                vid = f"ARCH{len(out)+1:03d}"

            out.append(Variant(vid=vid, source_symbol=src, genome=genome, notes=notes))
        except Exception:
            continue

    # Ensure unique IDs
    used = set()
    fixed: List[Variant] = []
    for i, v in enumerate(out, start=1):
        vid = v.vid or f"ARCH{i:03d}"
        if vid in used:
            vid = f"{vid}_{i}"
        used.add(vid)
        fixed.append(Variant(vid=vid, source_symbol=v.source_symbol, genome=v.genome, notes=v.notes))

    return fixed


def run_architect_queue_backtest(
    *,
    db_manager: Any,
    config: Any,
    variants: List[Variant],
    symbols: List[str],
    history_limit: int = 3000,
    max_workers: int = 4,
    progress_cb: Optional[ProgressFn] = None,
    score_key: str = "score",
) -> Dict[str, Any]:
    """Run orchestrated backtests.

    Returns dict:
      - df: pandas DataFrame (or None)
      - variants: list of variants (serializable)
      - aggregates: per-variant aggregates + best counts
      - meta: run metadata
    """

    if pd is None:
        return {"df": None, "variants": [], "aggregates": {}, "meta": {"error": "pandas_not_available"}}

    symbols = [str(s).strip() for s in (symbols or []) if str(s).strip()]
    variants = list(variants or [])

    ts = datetime.now(timezone.utc).replace(microsecond=0)

    if not variants:
        return {
            "df": pd.DataFrame(),
            "variants": [],
            "aggregates": {},
            "meta": {"exported_at_utc": ts.isoformat(), "symbols": len(symbols), "variants": 0},
        }

    if progress_cb:
        progress_cb(f"Orchestrator: evaluating {len(variants)} variants across {len(symbols)} symbols...", 0.0)

    def _eval_symbol(sym: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {"symbol": sym}
        try:
            arch = TheArchitect(db_manager, config=config)
            df = db_manager.get_history(sym, limit=int(history_limit))
            if df is None or getattr(df, "empty", False):
                # keep row but mark as no data
                row["best_variant"] = None
                row["best_profit"] = 0.0
                row["best_win_rate"] = 0.0
                row["best_trades"] = 0
                row["best_score"] = 0.0
                return row

            best_vid = None
            best_profit = None
            best_score = None
            best_wr = 0.0
            best_trades = 0

            for v in variants:
                res = arch.fitness_function(df, v.genome)

                try:
                    profit = float(res.get("profit", 0.0) or 0.0)
                except Exception:
                    profit = 0.0

                try:
                    trades = int(float(res.get("trades", 0) or 0))
                except Exception:
                    trades = 0

                try:
                    win_rate = float(res.get("win_rate", 0.0) or 0.0)
                except Exception:
                    win_rate = 0.0

                try:
                    score = float(res.get("score", profit) or profit)
                except Exception:
                    score = profit

                row[f"PL_{v.vid}"] = profit
                row[f"Trades_{v.vid}"] = trades
                row[f"WR_{v.vid}"] = win_rate
                row[f"Score_{v.vid}"] = score

                # choose best per symbol
                key_val = score if score_key == "score" else profit
                if best_vid is None:
                    best_vid = v.vid
                    best_profit = profit
                    best_score = score
                    best_wr = win_rate
                    best_trades = trades
                else:
                    if key_val > (best_score if score_key == "score" else best_profit):
                        best_vid = v.vid
                        best_profit = profit
                        best_score = score
                        best_wr = win_rate
                        best_trades = trades

            row["best_variant"] = best_vid
            row["best_profit"] = float(best_profit or 0.0)
            row["best_win_rate"] = float(best_wr or 0.0)
            row["best_trades"] = int(best_trades or 0)
            row["best_score"] = float(best_score or 0.0)
            return row

        except Exception:
            row["best_variant"] = None
            row["best_profit"] = 0.0
            row["best_win_rate"] = 0.0
            row["best_trades"] = 0
            row["best_score"] = 0.0
            return row

    rows: List[Dict[str, Any]] = []

    # Keep concurrency modest: DB access is locked in split mode.
    workers = max(1, min(int(max_workers or 1), 8, len(symbols) or 1))

    done = 0
    total = max(1, len(symbols))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_eval_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(f"Orchestrator progress: {done}/{len(symbols)}", done / total)

    # Stable ordering: alpha by symbol
    try:
        rows.sort(key=lambda r: str(r.get("symbol") or ""))
    except Exception:
        pass

    df_out = pd.DataFrame(rows)
    try:
        df_out["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # Aggregates
    best_counts: Dict[str, int] = {}
    per_variant: Dict[str, Dict[str, Any]] = {}

    for v in variants:
        vid = v.vid
        pl_col = f"PL_{vid}"
        tr_col = f"Trades_{vid}"
        wr_col = f"WR_{vid}"

        try:
            profits = df_out[pl_col].fillna(0.0).astype(float)
        except Exception:
            profits = None

        try:
            trades = df_out[tr_col].fillna(0).astype(float)
        except Exception:
            trades = None

        try:
            win_rates = df_out[wr_col].fillna(0.0).astype(float)
        except Exception:
            win_rates = None

        if profits is not None:
            pos = int((profits > 0).sum())
            n = int(len(profits))
            per_variant[vid] = {
                "symbols": n,
                "symbols_positive": pos,
                "symbol_win_rate": (pos / n) if n else None,
                "profit_sum": float(profits.sum()),
                "profit_mean": float(profits.mean()) if n else None,
                "trades_sum": int(trades.sum()) if trades is not None else None,
                "win_rate_mean": float(win_rates.mean()) if win_rates is not None else None,
            }
        else:
            per_variant[vid] = {
                "symbols": 0,
                "symbols_positive": 0,
                "symbol_win_rate": None,
                "profit_sum": 0.0,
                "profit_mean": None,
                "trades_sum": None,
                "win_rate_mean": None,
            }

    try:
        for x in df_out["best_variant"].fillna("None").astype(str).tolist():
            best_counts[x] = best_counts.get(x, 0) + 1
    except Exception:
        pass

    meta = {
        "exported_at_utc": ts.isoformat(),
        "symbols": len(symbols),
        "variants": len(variants),
        "history_limit": int(history_limit),
        "max_workers": int(workers),
        "score_key": str(score_key),
    }

    return {
        "df": df_out,
        "variants": [
            {"id": v.vid, "source_symbol": v.source_symbol, "genome": v.genome, "notes": v.notes}
            for v in variants
        ],
        "aggregates": {"best_variant_counts": best_counts, "per_variant": per_variant},
        "meta": meta,
    }
