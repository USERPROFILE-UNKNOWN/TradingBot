"""Chart rendering policies (LOCKED defaults).

v5.2.0: policy locking pass.
- Centralizes all tunables for Inspector chart rendering in a single immutable ChartPolicy.
- Default policy is intentionally chosen to reproduce v5.1.0 behavior:
  * y-limits derived from close quantiles (2%/98%) with 5% pad (fallback to min/max for small samples)
  * wick policy defaults to 'clip_to_ylim' when y-limits are provided
- Any future visual changes should be made by changing ChartPolicy defaults (explicitly) rather than ad-hoc edits in tabs.

NOTE: This module is intentionally *small and stable* to reduce regression risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Dict

XPolicy = Literal["datetime", "compressed_index"]
YPolicy = Literal["full", "quantile_close"]
WickPolicy = Literal["raw", "clip_to_ylim", "neutralize_outliers"]


@dataclass(frozen=True)
class ChartPolicy:
    """Immutable chart policy.

    Keep defaults stable unless intentionally changing visual behavior.
    """

    x_policy: XPolicy = "datetime"
    y_policy: YPolicy = "quantile_close"
    wick_policy: WickPolicy = "clip_to_ylim"

    # Quantile-close y-limit params (used when y_policy == 'quantile_close')
    quantile_lo: float = 0.02
    quantile_hi: float = 0.98
    pad_frac: float = 0.05
    min_points_for_quantile: int = 20


# Single source of truth: locked default policy.
DEFAULT_CHART_POLICY = ChartPolicy()


def get_locked_default_policy() -> ChartPolicy:
    """Return the locked default policy (explicit API)."""
    return DEFAULT_CHART_POLICY


def get_default_policies() -> Dict[str, Any]:
    """Back-compat helper for older call sites that expect a dict."""
    p = DEFAULT_CHART_POLICY
    return {"x_policy": p.x_policy, "y_policy": p.y_policy, "wick_policy": p.wick_policy}


def compute_ylim(df: Any, policy: ChartPolicy = DEFAULT_CHART_POLICY) -> Tuple[Optional[float], Optional[float]]:
    """Compute y-limits according to policy.

    Returns (None, None) if unable to compute.
    """
    try:
        if df is None or getattr(df, "empty", True):
            return None, None

        closes = df["close"].astype(float)
        if len(closes) == 0:
            return None, None

        if policy.y_policy == "full":
            lo = float(closes.min())
            hi = float(closes.max())
            if lo == hi:
                return lo - 1e-9, hi + 1e-9
            pad = max((hi - lo) * policy.pad_frac, 1e-9)
            return lo - pad, hi + pad

        # quantile_close (default)
        if len(closes) >= policy.min_points_for_quantile:
            q_lo = float(closes.quantile(policy.quantile_lo))
            q_hi = float(closes.quantile(policy.quantile_hi))
        else:
            q_lo = float(closes.min())
            q_hi = float(closes.max())

        if q_lo == q_hi:
            q_lo = float(closes.min())
            q_hi = float(closes.max())

        pad = max((q_hi - q_lo) * policy.pad_frac, 1e-9)
        return q_lo - pad, q_hi + pad

    except Exception:
        return None, None


def compute_quantile_close_ylim(df: Any) -> Tuple[Optional[float], Optional[float]]:
    """Back-compat: reproduce the v5.1.0 y-range behavior."""
    return compute_ylim(df, DEFAULT_CHART_POLICY)
