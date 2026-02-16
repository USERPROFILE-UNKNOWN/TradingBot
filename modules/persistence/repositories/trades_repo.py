from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import RepoBase


class TradesRepo(RepoBase):
    """Trade state + trade history repository.

    Uses:
      - active_trades.db (active_trades)
      - trade_history.db (trade_history)
    """
    def ensure_schema(self) -> None:
        """Ensure all trade-related tables exist.

        v5.16.2+: schema creation is owned by repositories (not database.py).
        """
        # active_trades.db
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        with lock:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS active_trades (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    entry_price REAL,
                    highest_price REAL,
                    strategy TEXT,
                    entry_time TEXT
                )
                """
            )
            conn.commit()

        # trade_history.db
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    strategy TEXT,
                    entry_time TEXT,
                    exit_time TEXT
                )
                """
            )
            # Indexes for common queries
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_history_entry_time ON trade_history(entry_time)")

            conn.commit()


    def get_trade_markers(self, symbol: str):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT entry_time, exit_time, entry_price, exit_price FROM trade_history WHERE symbol=?",
                    conn,
                    params=(symbol.upper(),),
                )
            except Exception:
                return None

        if df is None or df.empty:
            return None

        try:
            df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        except Exception:
            pass
        return df

    def log_trade_entry(self, symbol: str, qty: float, price: float, strategy: str):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO active_trades VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol.upper(), qty, price, price, strategy, datetime.now()),
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Trade Entry Error")
                except Exception:
                    pass

    def update_highest_price(self, symbol: str, new_high: float):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE active_trades SET highest_price = ? WHERE symbol = ?",
                    (new_high, symbol.upper()),
                )
                conn.commit()
            except Exception as e:
                # v5.12.3 updateA: don't silently swallow DB write failures.
                try:
                    from ...log_throttle import log_exception_throttled

                    log_exception_throttled(
                        self._log,
                        "E_DB_UPDATE_HIGHEST_PRICE",
                        e,
                        key=f"db_highest_{str(symbol).upper()}",
                        throttle_sec=300,
                        context={"symbol": symbol, "new_high": new_high},
                    )
                except Exception:
                    pass

    def get_active_trades_rows(self):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        """Return all active trades as raw DB rows for internal uses."""
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM active_trades")
            return cursor.fetchall()

    def get_active_trades(self):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        """Return dict of active trades for UI display."""
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM active_trades")
            trades = cursor.fetchall()

        # Convert to a dictionary for easier use
        active_trades = {}
        for t in trades:
            active_trades[t[0]] = {
                "qty": t[1],
                "entry_price": t[2],
                "highest_price": t[3],
                "strategy": t[4],
                "entry_time": t[5],
            }
        return active_trades

    def remove_active_trade(self, symbol: str):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        """Remove an active trade row without recording trade_history.
        Used for correcting DB state during syncs when a position does not exist.
        """
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Remove Active Trade Error")
                except Exception:
                    pass

    def close_trade(self, symbol: str, exit_price: float):
        if self.split_mode:
            with self._multi_lock(["active_trades", "trade_history"]):
                try:
                    at_conn = self._conn("active_trades")
                    th_conn = self._conn("trade_history")

                    at_cur = at_conn.cursor()
                    th_cur = th_conn.cursor()

                    at_cur.execute("SELECT * FROM active_trades WHERE symbol=?", (symbol.upper(),))
                    row = at_cur.fetchone()
                    if row:
                        qty = row[1]
                        entry_price = row[2]
                        strategy = row[4]
                        entry_time = row[5]
                        pl = (exit_price - entry_price) * qty

                        th_cur.execute(
                            """
                            INSERT INTO trade_history (symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (symbol.upper(), qty, entry_price, exit_price, pl, strategy, entry_time, datetime.now()),
                        )

                        at_cur.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))

                        th_conn.commit()
                        at_conn.commit()
                        return pl
                    return 0
                except Exception:
                    return 0

        # single-DB fallback (legacy)
        lock = self._lock("active_trades")
        conn = self._conn_fallback
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM active_trades WHERE symbol=?", (symbol.upper(),))
                row = cursor.fetchone()
                if row:
                    qty = row[1]
                    entry_price = row[2]
                    strategy = row[4]
                    entry_time = row[5]
                    pl = (exit_price - entry_price) * qty
                    cursor.execute(
                        """
                        INSERT INTO trade_history (symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (symbol.upper(), qty, entry_price, exit_price, pl, strategy, entry_time, datetime.now()),
                    )
                    cursor.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))
                    conn.commit()
                    return pl
                return 0
            except Exception:
                return 0

    def get_portfolio_stats(self):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        """Return portfolio stats as a dict used by the Portfolio tab."""
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(profit_loss) FROM trade_history")
            total_trades, total_profit = cursor.fetchone()
            total_trades = int(total_trades or 0)
            total_profit = float(total_profit or 0.0)
            cursor.execute("SELECT COUNT(*) FROM trade_history WHERE profit_loss > 0")
            wins = int((cursor.fetchone() or [0])[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM trade_history WHERE profit_loss < 0")
            losses = int((cursor.fetchone() or [0])[0] or 0)
            win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
            return {
                "total_trades": total_trades,
                "total_pl": total_profit,
                "wins": wins,
                "losses": losses,
                "win_rate": float(win_rate),
            }

    def get_strategy_stats(self):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT strategy, COUNT(*), SUM(profit_loss) FROM trade_history GROUP BY strategy")
            return cursor.fetchall()

    def get_recent_history(self, limit: int = 50):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time "
                "FROM trade_history ORDER BY exit_time DESC LIMIT ?",
                (int(limit),),
            )
            return cursor.fetchall()

    def get_trade_history_since(self, since_dt):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        """Return trade_history rows with exit_time >= since_dt.

        Args:
            since_dt: datetime or ISO string
        Returns:
            list[dict]
        """
        try:
            if isinstance(since_dt, datetime):
                since_str = since_dt.isoformat(sep=" ", timespec="seconds")
            else:
                since_str = str(since_dt)

            with lock:
                cur = conn.cursor()
                cur.execute(
                    "SELECT symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time "
                    "FROM trade_history WHERE exit_time >= ? ORDER BY exit_time ASC",
                    (since_str,),
                )
                rows = cur.fetchall()

            out = []
            for r in rows:
                out.append(
                    {
                        "symbol": r[0],
                        "qty": r[1],
                        "entry_price": r[2],
                        "exit_price": r[3],
                        "profit_loss": r[4],
                        "strategy": r[5],
                        "entry_time": r[6],
                        "exit_time": r[7],
                    }
                )
            return out
        except Exception:
            return []
