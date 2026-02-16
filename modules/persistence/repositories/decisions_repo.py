from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .base import RepoBase


class DecisionsRepo(RepoBase):

    def ensure_schema(self) -> None:
        with self._lock("decision_logs"):
            conn = self._conn("decision_logs")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    action TEXT,
                    price REAL,
                    rsi REAL,
                    ai_score REAL,
                    sentiment REAL,
                    reason TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_logs_ts ON decision_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol_ts ON decision_logs(symbol, timestamp)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_packets (
                    ts TEXT,
                    symbol TEXT,
                    packet_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_packets_ts ON decision_packets(ts)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_packets (
                    ts TEXT,
                    symbol TEXT,
                    packet_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_packets_ts ON execution_packets(ts)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config_history (
                    ts TEXT,
                    key TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    actor TEXT,
                    note TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_config_history_ts ON config_history(ts)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist_audit (
                    ts TEXT,
                    action TEXT,
                    symbol TEXT,
                    note TEXT,
                    actor TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_audit_ts ON watchlist_audit(ts)")
            conn.commit()

    """Decision logging repository.

    Uses:
      - decision_logs.db:
          - decision_logs (legacy quick log)
          - decision_packets / execution_packets (structured logging)
          - config_history / watchlist_audit (audit trail)
    """

    # --- legacy quick decision log (kept for backwards compatibility until v5.16.2 purge) ---

    def log_decision(self, symbol, strategy, action, price, rsi, ai_score, sentiment, reason):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO decision_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(), symbol.upper(), strategy, action, price, rsi, ai_score, sentiment, reason),
            )
            conn.commit()

    def get_recent_decisions(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT timestamp, symbol, strategy, action, price, rsi, ai_score, sentiment, reason "
                    "FROM decision_logs ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return df
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass
        return df

    def get_decision_counts_since(self, since_dt):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        """Return counts of decisions grouped by action since datetime."""
        try:
            if isinstance(since_dt, datetime):
                since_str = since_dt.isoformat(sep=" ", timespec="seconds")
            else:
                since_str = str(since_dt)
            with lock:
                cur = conn.cursor()
                cur.execute(
                    "SELECT action, COUNT(*) FROM decision_logs WHERE timestamp >= ? GROUP BY action",
                    (since_str,),
                )
                rows = cur.fetchall()
            out = {}
            for r in rows:
                if r and r[0] is not None:
                    out[str(r[0])] = int(r[1] or 0)
            return out
        except Exception:
            return {}

    # --- structured decision + execution packets ---

    def log_decision_packet(
        self,
        *,
        decision_id: str,
        symbol: str,
        strategy: str,
        action: str,
        score: float,
        price: float,
        ai_prob: float,
        sentiment: float,
        reason: str,
        market_regime: str,
        is_crypto: bool,
        payload: dict,
        timestamp=None,
    ) -> None:
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        ts = timestamp or datetime.now()
        try:
            pj = json.dumps(payload or {}, ensure_ascii=False)
        except Exception:
            pj = "{}"
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT OR REPLACE INTO decision_packets "
                    "(decision_id, timestamp, symbol, strategy, action, score, price, ai_prob, sentiment, reason, market_regime, is_crypto, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(decision_id),
                        ts,
                        str(symbol).upper(),
                        str(strategy),
                        str(action),
                        float(score),
                        float(price),
                        float(ai_prob),
                        float(sentiment),
                        str(reason),
                        str(market_regime),
                        1 if bool(is_crypto) else 0,
                        pj,
                    ),
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Decision Packet Error")
                except Exception:
                    pass

    def get_decision_packets(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT decision_id, timestamp, symbol, strategy, action, score, price, ai_prob, sentiment, reason, market_regime, is_crypto, payload_json "
                    "FROM decision_packets ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return df
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass
        return df

    def log_execution_packet(
        self,
        *,
        decision_id: str,
        symbol: str,
        status: str,
        requested_qty: float,
        filled_qty: float,
        avg_fill_price: float,
        order_id: str,
        error: str,
        payload: dict,
        timestamp=None,
    ) -> None:
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        ts = timestamp or datetime.now()
        try:
            pj = json.dumps(payload or {}, ensure_ascii=False)
        except Exception:
            pj = "{}"
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT OR REPLACE INTO execution_packets "
                    "(decision_id, timestamp, symbol, status, requested_qty, filled_qty, avg_fill_price, order_id, error, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(decision_id),
                        ts,
                        str(symbol).upper(),
                        str(status),
                        float(requested_qty),
                        float(filled_qty),
                        float(avg_fill_price),
                        str(order_id or ""),
                        str(error or ""),
                        pj,
                    ),
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Execution Packet Error")
                except Exception:
                    pass

    def get_execution_packets(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT decision_id, timestamp, symbol, status, requested_qty, filled_qty, avg_fill_price, order_id, error, payload_json "
                    "FROM execution_packets ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return df
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass
        return df

    # --- config/watchlist audits ---

    def log_config_change(self, section: str, key: str, old_value: str, new_value: str, source: str = "UI"):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO config_history (timestamp, section, key, old_value, new_value, source) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now(), str(section), str(key), str(old_value), str(new_value), str(source)),
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Config History Error")
                except Exception:
                    pass

    def log_watchlist_audit(self, action: str, symbol: str, source: str = "UI", note: str = ""):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO watchlist_audit (timestamp, action, symbol, source, note) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(), str(action), str(symbol).upper(), str(source), str(note)),
                )
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Watchlist Audit Error")
                except Exception:
                    pass

    def get_config_history(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT timestamp, section, key, old_value, new_value, source FROM config_history ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return df
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass
        return df
