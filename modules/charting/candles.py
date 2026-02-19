"""Candlestick drawing primitives.

v5.1.0: first wired usage (InspectorTab only). Goal is *zero behavior change*:
- Candles drawn with bar bodies and vlines for wicks.
- Wick clipping to y-limits is applied when y_lo/y_hi are provided.

v5.2.0: policy-locking support:
- Adds explicit wick_policy argument (default keeps prior behavior).
"""

from __future__ import annotations

from typing import Any, Optional, Literal

WickPolicy = Literal["raw", "clip_to_ylim", "neutralize_outliers"]


def draw_candles(
    ax: Any,
    df: Any,
    *,
    y_lo: Optional[float] = None,
    y_hi: Optional[float] = None,
    wick_policy: WickPolicy = "clip_to_ylim",
    width: float = 0.6,
    up_color: str = "#00C853",
    down_color: str = "#FF1744",
    wick_linewidth: float = 1.0,
) -> None:
    """Draw candlesticks onto a matplotlib axis.

    Assumes df has columns: open, high, low, close and uses df.index for x.
    """
    if df is None or getattr(df, "empty", True):
        return

    # Up/down partition
    up = df[df.close >= df.open]
    down = df[df.close < df.open]

    # Bodies
    ax.bar(up.index, up.close - up.open, width, bottom=up.open, color=up_color)
    ax.bar(down.index, down.open - down.close, width, bottom=down.close, color=down_color)

    # Wicks: explicit policy
    do_clip = (wick_policy in ("clip_to_ylim", "neutralize_outliers")) and (y_lo is not None) and (y_hi is not None)
    if do_clip:
        up_low = up.low.clip(lower=y_lo, upper=y_hi)
        up_high = up.high.clip(lower=y_lo, upper=y_hi)
        dn_low = down.low.clip(lower=y_lo, upper=y_hi)
        dn_high = down.high.clip(lower=y_lo, upper=y_hi)
    else:
        up_low = up.low
        up_high = up.high
        dn_low = down.low
        dn_high = down.high

    ax.vlines(up.index, up_low, up_high, color=up_color, linewidth=wick_linewidth)
    ax.vlines(down.index, dn_low, dn_high, color=down_color, linewidth=wick_linewidth)
