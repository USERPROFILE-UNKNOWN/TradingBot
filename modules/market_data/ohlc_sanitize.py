"""OHLC sanitization utilities.

This module centralizes *time-shape* normalization for chart consumers.

Responsibilities:
  - Ensure timestamps are valid & ascending.
  - Collapse multiple rows that fall within the same minute (common for some crypto feeds),
    while preserving OHLC semantics:
      open=first, high=max, low=min, close=last, volume=sum
  - For any extra numeric/indicator columns (RSI/BB/EMA/ADX/ATR/etc), keep the last value
    within that minute (best-effort alignment).
"""

from __future__ import annotations

from typing import Any


def sanitize_ohlc(df: Any) -> Any:
    """Normalize timestamps, sort, and handle same-minute duplicates.

    Returns a new DataFrame (or the original if sanitization isn't needed).
    """
    try:
        import pandas as pd
    except Exception:
        return df

    if df is None:
        return df
    try:
        if getattr(df, "empty", False):
            return df
    except Exception:
        return df

    if "timestamp" not in getattr(df, "columns", []):
        return df

    out = df.copy()

    try:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    except Exception:
        try:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        except Exception:
            return df

    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return out

    out = out.sort_values("timestamp").reset_index(drop=True)

    # Floor to minute
    try:
        ts_min = out["timestamp"].dt.floor("min")
    except Exception:
        return out

    # Fast path: already one row per minute
    try:
        if ts_min.is_unique:
            return out
    except Exception:
        pass

    # Aggregate within each minute
    agg = {}
    if "open" in out.columns:
        agg["open"] = "first"
    if "high" in out.columns:
        agg["high"] = "max"
    if "low" in out.columns:
        agg["low"] = "min"
    if "close" in out.columns:
        agg["close"] = "last"
    if "volume" in out.columns:
        agg["volume"] = "sum"

    # Preserve any extra columns (indicators, etc.) using last value of the minute
    for c in out.columns:
        if c in ("timestamp",):
            continue
        if c not in agg:
            agg[c] = "last"

    try:
        out2 = out.assign(_ts_min=ts_min).groupby("_ts_min", as_index=False).agg(agg)
        out2 = out2.rename(columns={"_ts_min": "timestamp"}).reset_index(drop=True)
        return out2
    except Exception:
        return out
