from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import RepoBase


class HistoryRepo(RepoBase):

    def ensure_schema(self) -> None:
        """Create required tables/indexes if missing."""
        with self._lock("historical_prices"):
            conn = self._conn("historical_prices")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_prices (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    close REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    volume REAL,
                    rsi REAL,
                    bb_lower REAL,
                    bb_upper REAL,
                    ema_200 REAL,
                    adx REAL,
                    atr REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_symbol_ts ON historical_prices(symbol, timestamp)")
            conn.commit()

    """Historical price store repository.

    Uses:
      - historical_prices.db (historical_prices)
    """

    def save_bulk_data(self, symbol, data_list):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Bulk insert minute bars into historical_prices.

        Returns the number of rows actually inserted (INSERT OR IGNORE semantics).
        """
        with lock:
            try:
                before = conn.total_changes
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO historical_prices 
                    (symbol, timestamp, close, open, high, low, volume, rsi, bb_lower, bb_upper, ema_200, adx, atr) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data_list,
                )
                conn.commit()
                after = conn.total_changes
                return max(0, after - before)
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Bulk Write Error")
                except Exception:
                    pass
                return 0

    def save_snapshot(
        self,
        symbol: str,
        timestamp: str,
        close: float,
        open_p: float,
        high: float,
        low: float,
        volume: float,
        rsi: float,
        bb_lower: float,
        bb_upper: float,
        ema_200: float,
        adx: float,
        atr: float,
    ):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO historical_prices 
                (symbol, timestamp, close, open, high, low, volume, rsi, bb_lower, bb_upper, ema_200, adx, atr) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol.upper(), timestamp, close, open_p, high, low, volume, rsi, bb_lower, bb_upper, ema_200, adx, atr),
            )
            conn.commit()

    def get_last_timestamp(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return the most recent timestamp stored for a symbol in historical_prices (UTC-aware datetime) or None."""
        sym_u = symbol.upper()
        sym_l = symbol.lower()
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM historical_prices WHERE symbol=?", (sym_u,))
            row = cursor.fetchone()
            if (not row) or row[0] is None:
                if sym_l != sym_u:
                    cursor.execute("SELECT MAX(timestamp) FROM historical_prices WHERE symbol=?", (sym_l,))
                    row = cursor.fetchone()
                if (not row) or row[0] is None:
                    return None
            ts = row[0]
        try:
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            try:
                if isinstance(ts, datetime):
                    return ts
            except Exception:
                pass
            return None

    def get_first_timestamp(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return the oldest timestamp stored for a symbol in historical_prices (UTC-aware datetime) or None."""
        sym_u = symbol.upper()
        sym_l = symbol.lower()
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(timestamp) FROM historical_prices WHERE symbol=?", (sym_u,))
            row = cursor.fetchone()
            if (not row) or row[0] is None:
                if sym_l != sym_u:
                    cursor.execute("SELECT MIN(timestamp) FROM historical_prices WHERE symbol=?", (sym_l,))
                    row = cursor.fetchone()
                if (not row) or row[0] is None:
                    return None
            ts = row[0]
        try:
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            try:
                if isinstance(ts, datetime):
                    return ts
            except Exception:
                pass
            return None

    def get_history(self, symbol: str, limit: int = 1000):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT * FROM historical_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(symbol.upper(), int(limit)),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return None
        # normalize types
        for col in ["open", "high", "low", "close", "volume", "rsi", "bb_lower", "bb_upper", "ema_200", "adx", "atr"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df

    def get_latest_snapshot(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return latest CLOSE price for the symbol (float) or None."""
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT close FROM historical_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
                (symbol.upper(),),
            )
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] is not None else None

    def get_all_symbols(self):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return distinct symbols stored in historical_prices."""
        with lock:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT symbol FROM historical_prices ORDER BY symbol")
            rows = cur.fetchall()
        return [r[0] for r in rows if r and r[0]]

    def get_latest_timestamps_for_symbols(self, symbols):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return dict mapping symbol->latest timestamp (as str) for given symbols."""
        out = {}
        try:
            sym_list = [s.upper() for s in symbols if s]
            if not sym_list:
                return out
            placeholders = ",".join(["?"] * len(sym_list))
            with lock:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT symbol, MAX(timestamp) FROM historical_prices WHERE symbol IN ({placeholders}) GROUP BY symbol",
                    tuple(sym_list),
                )
                rows = cur.fetchall()
            for r in rows:
                if r and r[0]:
                    out[str(r[0]).upper()] = r[1]
            return out
        except Exception:
            return out
