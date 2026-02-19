from __future__ import annotations

import json

import pandas as pd

from .base import RepoBase


class BacktestRepo(RepoBase):
    """Backtest results repository.

    Uses:
      - backtest_results.db (backtest_results)
    """

    def rebuild_backtest_table(self, strategies=None):
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
                        timeframe TEXT,
                        strategy TEXT,
                        start_date TEXT,
                        end_date TEXT,
                        win_rate REAL,
                        total_profit REAL,
                        max_drawdown REAL,
                        expectancy REAL,
                        sharpe_ratio REAL,
                        trade_count INTEGER,
                        best_params TEXT,
                        tested_at TEXT,
                        timestamp TEXT,
                        best_strategy TEXT,
                        best_profit REAL,
                        best_strategy_fingerprint TEXT,
                        strategy_fingerprints_json TEXT,
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
                "timeframe": "TEXT",
                "strategy": "TEXT",
                "start_date": "TEXT",
                "end_date": "TEXT",
                "win_rate": "REAL",
                "total_profit": "REAL",
                "max_drawdown": "REAL",
                "expectancy": "REAL",
                "sharpe_ratio": "REAL",
                "trade_count": "INTEGER",
                "best_params": "TEXT",
                "tested_at": "TEXT",
                "timestamp": "TEXT",
                "best_strategy": "TEXT",
                "best_profit": "REAL",
                "best_strategy_fingerprint": "TEXT",
                "strategy_fingerprints_json": "TEXT",
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

    @staticmethod
    def _pandas_available() -> bool:
        return hasattr(pd, "read_sql_query") and hasattr(pd, "DataFrame")

    class _MiniRow(dict):
        def to_dict(self):
            return dict(self)

    class _MiniILoc:
        def __init__(self, rows):
            self._rows = rows

        def __getitem__(self, index):
            return BacktestRepo._MiniRow(self._rows[index])

    class _MiniDataFrame:
        def __init__(self, rows):
            self._rows = list(rows or [])
            self.iloc = BacktestRepo._MiniILoc(self._rows)

        @property
        def empty(self):
            return len(self._rows) == 0

        @property
        def columns(self):
            if not self._rows:
                return []
            return list(self._rows[0].keys())

    def _empty_df(self):
        if self._pandas_available():
            return pd.DataFrame()
        return BacktestRepo._MiniDataFrame([])

    def _read_sql_query(self, query: str, conn, params=None):
        if self._pandas_available():
            if params is None:
                return pd.read_sql_query(query, conn)
            return pd.read_sql_query(query, conn, params=params)

        cur = conn.cursor()
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        as_dicts = [dict(zip(cols, row)) for row in rows]
        return BacktestRepo._MiniDataFrame(as_dicts)


    def save_backtest_result(self, data_dict):
        try:
            self.ensure_backtest_table()
        except Exception:
            pass

        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(backtest_results)")
                existing_cols = {str(r[1]) for r in cursor.fetchall()}

                payload = dict(data_dict or {})

                # v6.16.2: normalize common full-backtest fields into stable legacy columns
                # so UI/export paths that still read these columns do not see all NULLs.
                best_strategy = payload.get("best_strategy")
                if best_strategy is not None and "strategy" not in payload:
                    payload["strategy"] = best_strategy

                best_profit = payload.get("best_profit")
                if best_profit is not None and "total_profit" not in payload:
                    payload["total_profit"] = best_profit

                ts = payload.get("tested_at") or payload.get("timestamp")
                if ts is not None:
                    payload.setdefault("tested_at", ts)
                    payload.setdefault("timestamp", ts)

                payload.setdefault("timeframe", "1Min")
                payload.setdefault("expectancy", payload.get("avg_pl"))

                if "results_json" not in payload:
                    compact = {}
                    for k, v in payload.items():
                        ks = str(k)
                        if ks.startswith("PL_") or ks.startswith("Trades_"):
                            compact[ks] = v
                    if compact:
                        try:
                            payload["results_json"] = json.dumps(compact, ensure_ascii=False)
                        except Exception:
                            payload["results_json"] = None

                # full backtest rows may include dynamic strategy columns
                for key, value in payload.items():
                    if key in existing_cols:
                        continue

                    val = value
                    if isinstance(val, bool):
                        ctype = "INTEGER"
                    elif isinstance(val, int):
                        ctype = "INTEGER"
                    elif isinstance(val, float):
                        ctype = "REAL"
                    else:
                        ctype = "TEXT"

                    try:
                        cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {key} {ctype}")
                        existing_cols.add(str(key))
                    except Exception:
                        pass

                filtered = {k: v for k, v in payload.items() if k in existing_cols}
                if not filtered:
                    return

                columns = ', '.join(filtered.keys())
                placeholders = ', '.join(['?'] * len(filtered))
                sql = f"INSERT OR REPLACE INTO backtest_results ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, list(filtered.values()))
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
                        return self._read_sql_query(
                            "SELECT * FROM backtest_results WHERE symbol=? ORDER BY tested_at DESC",
                            conn,
                            params=(symbol.upper(),),
                        )
                    except Exception:
                        # Older DBs may not have tested_at.
                        try:
                            return self._read_sql_query(
                                "SELECT * FROM backtest_results WHERE symbol=? ORDER BY timestamp DESC",
                                conn,
                                params=(symbol.upper(),),
                            )
                        except Exception:
                            return self._read_sql_query(
                                "SELECT * FROM backtest_results WHERE symbol=?",
                                conn,
                                params=(symbol.upper(),),
                            )
                else:
                    try:
                        return self._read_sql_query("SELECT * FROM backtest_results ORDER BY tested_at DESC", conn)
                    except Exception:
                        try:
                            return self._read_sql_query("SELECT * FROM backtest_results ORDER BY timestamp DESC", conn)
                        except Exception:
                            return self._read_sql_query("SELECT * FROM backtest_results", conn)
            except Exception:
                return self._empty_df()

    def get_best_strategy_for_symbol(self, symbol: str):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                df = self._read_sql_query(
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
