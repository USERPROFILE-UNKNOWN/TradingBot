from __future__ import annotations

import pandas as pd

from .base import RepoBase


class BacktestRepo(RepoBase):
    """Backtest results repository.

    Uses:
      - backtest_results.db (backtest_results)
    """

    def rebuild_backtest_table(self):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        """Drop and recreate backtest_results table (legacy columns supported)."""
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("DROP VIEW IF EXISTS backtest_testing_data")
                cursor.execute("DROP TABLE IF EXISTS backtest_results")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        strategy TEXT,
                        start_date TEXT,
                        end_date TEXT,
                        win_rate REAL,
                        total_profit REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        trade_count INTEGER,
                        best_params TEXT,
                        tested_at TEXT,
                        results_json TEXT,
                        UNIQUE(symbol, strategy, start_date, end_date)
                    )
                    """
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Error rebuilding backtest_results table")
                except Exception:
                    pass

    def ensure_backtest_table(self):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        """Ensure backtest_results table exists with required columns.

        This method is forward-only additive (no destructive schema changes).
        """
        with lock:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS backtest_results (id INTEGER PRIMARY KEY AUTOINCREMENT)")
            conn.commit()

            cur.execute("PRAGMA table_info(backtest_results)")
            cols = {r[1] for r in cur.fetchall()}

            required = {
                "symbol": "TEXT",
                "strategy": "TEXT",
                "start_date": "TEXT",
                "end_date": "TEXT",
                "win_rate": "REAL",
                "total_profit": "REAL",
                "max_drawdown": "REAL",
                "sharpe_ratio": "REAL",
                "trade_count": "INTEGER",
                "best_params": "TEXT",
                "tested_at": "TEXT",
                "results_json": "TEXT",
            }

            for c, ctype in required.items():
                if c not in cols:
                    try:
                        cur.execute(f"ALTER TABLE backtest_results ADD COLUMN {c} {ctype}")
                    except Exception:
                        pass

            # Unique index if possible
            try:
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_backtest_unique ON backtest_results(symbol, strategy, start_date, end_date)"
                )
            except Exception:
                pass

            # Secondary indexes for UI queries
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_tested_at ON backtest_results(tested_at)")
            except Exception:
                pass

            conn.commit()
    def ensure_schema(self) -> None:
        """Ensure all backtest-related tables/indexes exist.

        v5.16.2+: schema creation is owned by repositories (not database.py).
        """
        self.ensure_backtest_table()


    def save_backtest_result(self, data_dict):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                cursor = conn.cursor()
                columns = ', '.join(data_dict.keys())
                placeholders = ', '.join(['?'] * len(data_dict))
                sql = f"INSERT OR REPLACE INTO backtest_results ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, list(data_dict.values()))
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Error Saving Backtest")
                except Exception:
                    pass

    def get_backtest_data(self, symbol: str = None):
        """Return backtest results as a DataFrame.

        Legacy behavior: callers expect a DataFrame (possibly empty), never None.
        """
        # Ensure schema first (acquires the same per-DB lock).
        try:
            self.ensure_backtest_table()
        except Exception:
            pass

        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:

            try:
                if symbol:
                    try:
                        return pd.read_sql_query(
                            "SELECT * FROM backtest_results WHERE symbol=? ORDER BY tested_at DESC",
                            conn,
                            params=(symbol.upper(),),
                        )
                    except Exception:
                        # Older DBs may not have tested_at.
                        try:
                            return pd.read_sql_query(
                                "SELECT * FROM backtest_results WHERE symbol=? ORDER BY timestamp DESC",
                                conn,
                                params=(symbol.upper(),),
                            )
                        except Exception:
                            return pd.read_sql_query(
                                "SELECT * FROM backtest_results WHERE symbol=?",
                                conn,
                                params=(symbol.upper(),),
                            )
                else:
                    try:
                        return pd.read_sql_query("SELECT * FROM backtest_results ORDER BY tested_at DESC", conn)
                    except Exception:
                        try:
                            return pd.read_sql_query("SELECT * FROM backtest_results ORDER BY timestamp DESC", conn)
                        except Exception:
                            return pd.read_sql_query("SELECT * FROM backtest_results", conn)
            except Exception:
                return pd.DataFrame()

    def get_best_strategy_for_symbol(self, symbol: str):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                df = pd.read_sql_query(
                    """
                    SELECT strategy, win_rate, total_profit, max_drawdown, sharpe_ratio
                    FROM backtest_results
                    WHERE symbol=?
                    ORDER BY total_profit DESC
                    LIMIT 1
                    """,
                    conn,
                    params=(symbol.upper(),),
                )
                if df is None or df.empty:
                    return None
                row = df.iloc[0].to_dict()
                return row
            except Exception:
                return None
