"""Headless full backtest service.

v6.7.0:
- Extracts full backtest orchestration from UI so both UI and AgentMaster can call
  the same service path.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional



def _strategy_params(config: Any, strategy_name: str) -> Dict[str, str]:
    sec = f"STRATEGY_{strategy_name}"
    try:
        if hasattr(config, "has_section") and config.has_section(sec):
            return {str(k): str(v) for k, v in config[sec].items()}
    except Exception:
        pass
    return {}


def _strategy_fingerprint(config: Any, strategy_name: str) -> str:
    payload = _strategy_params(config, strategy_name)
    try:
        b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        b = repr(payload).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def run_full_backtest_service(
    config: Any,
    db_manager: Any,
    *,
    simulate_strategy: Optional[Callable[[Any, str, str], tuple[float, int]]] = None,
    log: Optional[Callable[[str], None]] = None,
    symbols: Optional[Iterable[str]] = None,
    rebuild_table: bool = True,
    sleep_per_symbol_sec: float = 0.0,
) -> Dict[str, Any]:
    """Run a full symbol x strategy backtest sweep and persist compact results."""
    logger = log or (lambda *_a, **_k: None)

    if simulate_strategy is None:
        from .quick_backtest import simulate_strategy_numpy as _quick_simulate_strategy_numpy

        def _default_sim(df: Any, strat: str, _symbol: str) -> tuple[float, int]:
            sec = f"STRATEGY_{strat}" if not str(strat).upper().startswith("STRATEGY_") else str(strat)
            total_pl, trades, _win_rate, _avg_pl, _max_dd = _quick_simulate_strategy_numpy(df, config, sec)
            return float(total_pl), int(trades)

        simulate_strategy = _default_sim

    strategies = [s.replace("STRATEGY_", "") for s in config.sections() if str(s).startswith("STRATEGY_")]
    if not strategies:
        return {"ok": False, "reason": "no_strategies", "count": 0}

    strategy_fingerprints = {s: _strategy_fingerprint(config, s) for s in strategies}
    by_fp: Dict[str, List[str]] = {}
    for sn, fp in strategy_fingerprints.items():
        by_fp.setdefault(fp, []).append(sn)
    duplicate_fingerprint_groups = [sorted(v) for v in by_fp.values() if len(v) > 1]
    if duplicate_fingerprint_groups:
        logger(f"[BACKTEST] ⚠ Strategy fingerprint collisions detected: {duplicate_fingerprint_groups}")

    if rebuild_table:
        rebuilt = False
        try:
            db_manager.rebuild_backtest_table(strategies)
            rebuilt = True
        except TypeError:
            # v6.16.2: support repository implementations that expose rebuild without args.
            try:
                db_manager.rebuild_backtest_table()
                rebuilt = True
            except Exception as e:
                logger(f"[BACKTEST] ⚠ rebuild_backtest_table fallback failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger(f"[BACKTEST] ⚠ rebuild_backtest_table failed: {type(e).__name__}: {e}")

        if not rebuilt:
            try:
                db_manager.ensure_backtest_table()
            except Exception as e:
                logger(f"[BACKTEST] ⚠ ensure_backtest_table failed after rebuild failure: {type(e).__name__}: {e}")

    if symbols is None:
        try:
            symbols = db_manager.get_all_symbols() or []
        except Exception:
            symbols = []

    syms = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]
    if not syms:
        return {"ok": False, "reason": "no_symbols", "count": 0}

    count = 0
    saved_count = 0
    save_failures = 0
    run_rows: List[Dict[str, Any]] = []

    for sym in syms:
        try:
            df = db_manager.get_history(sym, 5000)
        except Exception:
            df = None

        res: Dict[str, Any] = {"symbol": sym}
        res["timeframe"] = "1Min"
        res["start_date"] = None
        res["end_date"] = None
        best_strat = "None"
        best_profit = -999999.0

        try:
            if df is not None and not getattr(df, "empty", True):
                idx = getattr(df, "index", None)
                if idx is not None and len(idx) > 0:
                    res["start_date"] = str(idx[0])
                    res["end_date"] = str(idx[-1])
        except Exception:
            pass

        for s in strategies:
            try:
                if df is not None and not getattr(df, "empty", True):
                    pl, trades = simulate_strategy(df, s, sym)
                    pl = float(pl)
                    trades = int(trades)
                else:
                    pl, trades = 0.0, 0
            except Exception:
                pl, trades = 0.0, 0

            res[f"PL_{s}"] = round(float(pl), 2)
            res[f"Trades_{s}"] = int(trades)
            if float(pl) > best_profit:
                best_profit = float(pl)
                best_strat = s

        res["best_strategy"] = best_strat
        res["best_profit"] = round(best_profit, 2) if best_profit != -999999.0 else 0.0
        res["best_strategy_fingerprint"] = strategy_fingerprints.get(best_strat, "") if best_strat != "None" else ""
        try:
            res["strategy_fingerprints_json"] = json.dumps(strategy_fingerprints, sort_keys=True)
        except Exception:
            pass
        try:
            res["trade_count"] = int(res.get(f"Trades_{best_strat}", 0) or 0) if best_strat != "None" else 0
        except Exception:
            res["trade_count"] = 0
        res["win_rate"] = None
        res["max_drawdown"] = None
        res["expectancy"] = None
        try:
            matrix = {k: v for k, v in res.items() if str(k).startswith("PL_") or str(k).startswith("Trades_")}
            res["results_json"] = json.dumps(matrix, ensure_ascii=False)
        except Exception:
            pass
        res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_rows.append(dict(res))

        if sleep_per_symbol_sec > 0:
            time.sleep(max(0.0, float(sleep_per_symbol_sec)))

        try:
            db_manager.save_backtest_result(res)
            saved_count += 1
        except Exception:
            save_failures += 1

        count += 1
        if count % 5 == 0:
            logger(f"Backtesting... {count}/{len(syms)}")

    logger("✅ Backtest Complete.")
    return {
        "ok": True,
        "count": count,
        "symbols": len(syms),
        "strategies": len(strategies),
        "saved_count": saved_count,
        "save_failures": save_failures,
        "rows": run_rows,
        "strategy_fingerprints": strategy_fingerprints,
        "duplicate_fingerprint_groups": duplicate_fingerprint_groups,
    }
