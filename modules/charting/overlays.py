"""Chart overlays.

v5.1.0: first wired usage (InspectorTab only). Goal is *zero behavior change*:
- Lower Bollinger Band (bb_lower) cyan dashed
- EMA 200 (ema_200) yellow
- ADX on twinx axis (0-100) white
"""

from __future__ import annotations

from typing import Any, Optional

def draw_lower_band(ax: Any, df: Any, *, color: str = "cyan") -> None:
    if df is None or getattr(df, "empty", True):
        return
    if "bb_lower" in df.columns:
        ax.plot(df.index, df["bb_lower"], color=color, linestyle="--", alpha=0.5, label="Lower Band")

def draw_ema200(ax: Any, df: Any, *, color: str = "yellow") -> None:
    if df is None or getattr(df, "empty", True):
        return
    if "ema_200" in df.columns:
        ax.plot(df.index, df["ema_200"], color=color, linewidth=1, label="200 EMA")

def draw_adx(ax: Any, df: Any, *, color: str = "white") -> Optional[Any]:
    """Draw ADX on a secondary y-axis. Returns ax2 or None."""
    if df is None or getattr(df, "empty", True):
        return None
    if "adx" not in df.columns:
        return None

    ax2 = ax.twinx()
    ax2.plot(df.index, df["adx"], color=color, linewidth=1, alpha=0.7, label="ADX")
    ax2.tick_params(axis="y", colors="white")
    ax2.set_ylabel("ADX", color="white")
    ax2.set_ylim(0, 100)
    return ax2
