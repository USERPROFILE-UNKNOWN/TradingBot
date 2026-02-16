"""History repository.

In the v5.x refactor series, the Inspector and other consumers fetch OHLCV via
this layer. The purpose is to enforce a stable dataframe contract and
centralize validation/sanitization so chart behavior doesn't accidentally
change when unrelated parts of the codebase are touched.

Contract (v5.3.0):
  - Returns a pandas DataFrame sorted by ascending UTC timestamp.
  - Ensures `timestamp` exists and is datetime64[ns, UTC] where possible.
  - Collapses same-minute duplicates (best effort) via `ohlc_sanitize`.
  - Drops clearly invalid OHLC bars via `ohlc_validate`.
"""

from __future__ import annotations

from typing import Any

from .ohlc_sanitize import sanitize_ohlc
from .ohlc_validate import validate_ohlc


class HistoryRepo:
    """Thin repository wrapper around the DB manager.

    `db_manager` is expected to expose `get_history(symbol, limit=...)`.
    """

    def __init__(self, db_manager: Any):
        self.db_manager = db_manager

    def get_history(self, symbol: str, limit: int = 1000) -> Any:
        """Return canonical OHLCV history for `symbol`.

        Args:
            symbol: Symbol string (case-insensitive).
            limit: Target number of bars for consumers.

        Notes:
            We keep this layer conservative: it should not introduce new
            computed columns or indicators. It only normalizes/filters.
        """
        try:
            sym = str(symbol).strip().upper()
        except Exception:
            sym = symbol

        # Pull raw history from DB
        try:
            df = self.db_manager.get_history(sym, limit=int(limit))
        except Exception:
            return None

        # Canonicalize time-shape & validate OHLC
        df = sanitize_ohlc(df)
        df = validate_ohlc(df)

        # Cap to the last `limit` bars (ascending)
        try:
            if limit and getattr(df, 'shape', (0, 0))[0] > int(limit):
                df = df.tail(int(limit)).reset_index(drop=True)
        except Exception:
            pass

        return df
