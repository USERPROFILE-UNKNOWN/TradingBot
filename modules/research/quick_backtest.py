"""
modules/research/quick_backtest.py

Quick sanity backtest utilities (stdlib + numpy/pandas only).

Used by TradingView autovalidation (PAPER-only) to run a lightweight
strategy replay over the last N days and return a compact result bundle.

This intentionally duplicates (and keeps compatible with) the backtest
math used in UI.simulate_strategy_numpy, but is written as a pure module
to keep the worker path EXE-friendly and non-UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    total_pl: float
    trades: int
    win_rate: float
    avg_pl: float
    max_drawdown: float


def _cfg_get(config, section: str, key: str, fallback: Any = None) -> Any:
    try:
        if config.has_option(section, key):
            return config.get(section, key)
    except Exception:
        pass
    return fallback


def _cfg_getfloat(config, section: str, key: str, fallback: float) -> float:
    v = _cfg_get(config, section, key, None)
    if v is None:
        return float(fallback)
    try:
        return float(v)
    except Exception:
        return float(fallback)


def _cfg_getint(config, section: str, key: str, fallback: int) -> int:
    v = _cfg_get(config, section, key, None)
    if v is None:
        return int(fallback)
    try:
        return int(float(v))
    except Exception:
        return int(fallback)


def _cfg_getbool(config, section: str, key: str, fallback: bool) -> bool:
    v = _cfg_get(config, section, key, None)
    if v is None:
        return bool(fallback)
    try:
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    except Exception:
        pass
    return bool(fallback)


def select_champion_strategies(config, max_strategies: int = 6) -> List[str]:
    """
    "Champion strategy set" (fast) heuristic:
      - all STRATEGY_* sections
      - enabled == true (default true)
      - sorted by priority (default 0) descending
      - take up to max_strategies
    """
    sections = []
    try:
        for sec in config.sections():
            if sec.upper().startswith("STRATEGY_"):
                enabled = _cfg_getbool(config, sec, "enabled", True)
                if not enabled:
                    continue
                priority = _cfg_getint(config, sec, "priority", 0)
                sections.append((priority, sec))
    except Exception:
        return []

    sections.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in sections[: max(1, int(max_strategies))]]


def simulate_strategy_numpy(
    df: pd.DataFrame,
    config,
    strat_name: str,
    initial_capital: float = 10000.0,
) -> Tuple[float, int, float, float, float]:
    """
    Returns:
      total_pl, trades, win_rate, avg_pl, max_drawdown

    Assumes df contains:
      - close, low
      - rsi, bb_lower, ema_200, atr
      - timestamp (optional; not required for math)
    """

    # Defensive copy of needed columns only
    required = ["close", "low", "rsi", "bb_lower", "ema_200", "atr"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for quick backtest")

    close = df["close"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    rsi = df["rsi"].to_numpy(dtype=np.float64)
    bb_lower = df["bb_lower"].to_numpy(dtype=np.float64)
    ema_200 = df["ema_200"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)

    # Strategy parameters (match UI defaults)
    rsi_entry = _cfg_getfloat(config, strat_name, "rsi_entry", 35)
    rsi_exit = _cfg_getfloat(config, strat_name, "rsi_exit", 55)
    bb_proximity = _cfg_getfloat(config, strat_name, "bb_proximity", 1.01)
    ema_trend_filter = _cfg_getbool(config, strat_name, "ema_trend_filter", True)
    atr_multiplier_sl = _cfg_getfloat(config, strat_name, "atr_multiplier_sl", 1.5)
    atr_multiplier_tp = _cfg_getfloat(config, strat_name, "atr_multiplier_tp", 3.0)
    max_hold_bars = _cfg_getint(config, strat_name, "max_hold_bars", 240)

    # Guard rails (avoid weird configs)
    max_hold_bars = max(5, min(max_hold_bars, 5000))
    bb_proximity = max(0.5, min(bb_proximity, 2.0))
    atr_multiplier_sl = max(0.1, min(atr_multiplier_sl, 10.0))
    atr_multiplier_tp = max(0.1, min(atr_multiplier_tp, 20.0))

    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    capital = float(initial_capital)
    trades = 0

    win_count = 0
    trade_pls: List[float] = []

    equity_peak = capital
    max_drawdown = 0.0

    hold_bars = 0

    # Start after warmup bars to reduce noisy NaNs
    start_idx = 50
    n = len(close)
    if n <= start_idx + 5:
        return 0.0, 0, 0.0, 0.0, 0.0

    for i in range(start_idx, n):
        c = close[i]
        l = low[i]
        r = rsi[i]
        bb = bb_lower[i]
        ema = ema_200[i]
        a = atr[i]

        # Skip NaN rows defensively
        if np.isnan(c) or np.isnan(l) or np.isnan(r) or np.isnan(bb) or np.isnan(ema) or np.isnan(a):
            continue

        if in_trade:
            hold_bars += 1

            # Exit checks
            exit_reason = None
            exit_price = None

            if l <= stop_loss:
                exit_reason = "SL"
                exit_price = stop_loss
            elif c >= take_profit:
                exit_reason = "TP"
                exit_price = take_profit
            elif r >= rsi_exit:
                exit_reason = "RSI"
                exit_price = c
            elif hold_bars >= max_hold_bars:
                exit_reason = "TIME"
                exit_price = c

            if exit_reason is not None and exit_price is not None:
                pct_pl = (exit_price - entry_price) / entry_price
                pl = capital * pct_pl
                capital += pl

                trades += 1
                trade_pls.append(pl)
                if pl > 0:
                    win_count += 1

                in_trade = False
                entry_price = 0.0
                stop_loss = 0.0
                take_profit = 0.0
                hold_bars = 0

                # Equity + drawdown tracking
                equity_peak = max(equity_peak, capital)
                dd = (equity_peak - capital) / equity_peak if equity_peak > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)

            continue

        # Entry checks
        near_bb = c <= (bb * bb_proximity)
        rsi_ok = r <= rsi_entry
        trend_ok = (c >= ema) if ema_trend_filter else True

        if near_bb and rsi_ok and trend_ok:
            entry_price = c
            stop_loss = c - (a * atr_multiplier_sl)
            take_profit = c + (a * atr_multiplier_tp)
            in_trade = True
            hold_bars = 0

    total_pl = capital - float(initial_capital)
    win_rate = (win_count / trades) if trades > 0 else 0.0
    avg_pl = (sum(trade_pls) / trades) if trades > 0 else 0.0

    return float(total_pl), int(trades), float(win_rate), float(avg_pl), float(max_drawdown)


def run_quick_backtest(
    config,
    db_manager,
    symbol: str,
    days: int = 14,
    max_strategies: int = 6,
    min_trades: int = 1,
) -> Dict[str, Any]:
    """
    Fetch history for symbol from DB, slice last `days`, run champion strategies,
    and return a JSON-serializable bundle.

    Does NOT write to DB directly.
    """
    days = max(1, min(int(days), 90))
    max_strategies = max(1, min(int(max_strategies), 20))
    min_trades = max(0, min(int(min_trades), 50))

    # Pull enough bars then filter by time window.
    df = db_manager.get_history(symbol, limit=20000)
    if df is None or len(df) == 0:
        return {
            "ok": False,
            "symbol": symbol,
            "error": "No history found in DB",
            "results": [],
        }

    # Ensure timestamp is tz-aware UTC
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        if "timestamp" in df.columns:
            df = df[df["timestamp"] >= cutoff]
    except Exception:
        # If filtering fails, keep full df
        pass

    if len(df) < 100:
        return {
            "ok": False,
            "symbol": symbol,
            "error": f"Insufficient bars for quick backtest (rows={len(df)})",
            "results": [],
        }

    strategies = select_champion_strategies(config, max_strategies=max_strategies)
    if not strategies:
        return {
            "ok": False,
            "symbol": symbol,
            "error": "No enabled STRATEGY_* sections found",
            "results": [],
        }

    results: List[StrategyResult] = []
    errors: Dict[str, str] = {}

    for strat in strategies:
        try:
            total_pl, trades, win_rate, avg_pl, max_dd = simulate_strategy_numpy(df, config, strat)
            results.append(
                StrategyResult(
                    strategy=strat,
                    total_pl=total_pl,
                    trades=trades,
                    win_rate=win_rate,
                    avg_pl=avg_pl,
                    max_drawdown=max_dd,
                )
            )
        except Exception as e:
            errors[strat] = str(e)

    # Choose best by P/L among strategies with >= min_trades
    eligible = [r for r in results if r.trades >= min_trades]
    best = max(eligible, key=lambda r: r.total_pl, default=None)

    return {
        "ok": True,
        "symbol": symbol,
        "days": days,
        "max_strategies": max_strategies,
        "min_trades": min_trades,
        "results": [
            {
                "strategy": r.strategy,
                "total_pl": r.total_pl,
                "trades": r.trades,
                "win_rate": r.win_rate,
                "avg_pl": r.avg_pl,
                "max_drawdown": r.max_drawdown,
            }
            for r in results
        ],
        "best": None
        if best is None
        else {
            "strategy": best.strategy,
            "total_pl": best.total_pl,
            "trades": best.trades,
            "win_rate": best.win_rate,
            "avg_pl": best.avg_pl,
            "max_drawdown": best.max_drawdown,
        },
        "errors": errors,
    }
