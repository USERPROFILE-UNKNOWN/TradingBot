"""OHLC validation rules.

The goal of validation is to ensure that any dataframe handed to charting
code has a stable and safe contract.

Rules implemented (v5.3.0):
  - Require a usable timestamp column.
  - Coerce OHLC columns to numeric.
  - Drop rows with NaNs in open/high/low/close.
  - Drop rows with non-positive prices.
  - Drop rows where high < low.

We intentionally keep this conservative (drop clearly invalid bars only) to
avoid mutating or "correcting" valid market data.
"""

from __future__ import annotations

from typing import Any


def validate_ohlc(df: Any) -> Any:
    """Validate and filter invalid OHLC rows.

    Args:
        df: Pandas DataFrame-like object.
    Returns:
        A filtered/sanitized dataframe (or the original object if validation
        isn't possible).
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

    if 'timestamp' not in getattr(df, 'columns', []):
        return df

    out = df.copy()

    # Timestamp normalization
    try:
        out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce', utc=True)
    except Exception:
        try:
            out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
        except Exception:
            return df
    out = out.dropna(subset=['timestamp'])
    if out.empty:
        return out

    # Numeric coercion for OHLC
    for c in ('open', 'high', 'low', 'close'):
        if c in out.columns:
            try:
                out[c] = pd.to_numeric(out[c], errors='coerce')
            except Exception:
                pass

    # Drop invalid rows
    try:
        out = out.dropna(subset=['open', 'high', 'low', 'close'])
        out = out[(out['open'] > 0) & (out['high'] > 0) & (out['low'] > 0) & (out['close'] > 0)]
        out = out[out['high'] >= out['low']]
    except Exception:
        # If any required column is missing, just return the best effort frame.
        return out

    # Always sort ascending (chart contract)
    try:
        out = out.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        pass

    return out
