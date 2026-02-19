"""Headless full backtest service.

v6.7.0:
- Extracts full backtest orchestration from UI so both UI and AgentMaster can call
  the same service path.
"""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any, Callable, Dict, Iterable, List, Optional



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

    if rebuild_table:
        try:
            db_manager.rebuild_backtest_table(strategies)
        except Exception:
            # Keep running even if rebuild is unavailable in some environments.
            pass

    if symbols is None:
        try:
            symbols = db_manager.get_all_symbols() or []
        except Exception:
            symbols = []

    syms = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]
    if not syms:
        return {"ok": False, "reason": "no_symbols", "count": 0}

    count = 0
    for sym in syms:
        try:
            df = db_manager.get_history(sym, 5000)
        except Exception:
            df = None

        res: Dict[str, Any] = {"symbol": sym}
        best_strat = "None"
        best_profit = -999999.0

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
        res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if sleep_per_symbol_sec > 0:
            time.sleep(max(0.0, float(sleep_per_symbol_sec)))

        try:
            db_manager.save_backtest_result(res)
        except Exception:
            pass

        count += 1
        if count % 5 == 0:
            logger(f"Backtesting... {count}/{len(syms)}")

    logger("✅ Backtest Complete.")
    return {"ok": True, "count": count, "symbols": len(syms), "strategies": len(strategies)}
