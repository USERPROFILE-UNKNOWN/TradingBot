"""Metrics and audit persistence (metrics.db).

Phase 1.5 additions:
- tradingview_alerts table for durable TradingView webhook ingestion.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, Optional


class MetricsStore:
    def __init__(self, db_dir: str):
        self.path = os.path.join(db_dir, "metrics.db")
        os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    reason TEXT,
                    payload_json TEXT
                )
                """
            )

            # Phase 1.5: TradingView alert ingestion
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tradingview_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    symbol TEXT,
                    exchange TEXT,
                    timeframe TEXT,
                    signal TEXT,
                    price REAL,
                    raw_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tv_alerts_ts ON tradingview_alerts(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tv_alerts_processed ON tradingview_alerts(processed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tv_alerts_idem ON tradingview_alerts(idempotency_key)")


            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tradingview_backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    signal TEXT,
                    bundle_path TEXT,
                    best_strategy TEXT,
                    best_pl REAL,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tv_bt_ts ON tradingview_backtests(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tv_bt_symbol ON tradingview_backtests(symbol)")

            conn.commit()

    def log_metric(self, name: str, value: float, metadata: dict | None = None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO system_metrics(ts, metric_name, metric_value, metadata_json) VALUES (?, ?, ?, ?)",
                (int(time.time()), str(name), float(value), json.dumps(metadata or {})),
            )
            conn.commit()

    def log_agent_action(self, mode: str, action_type: str, approved: bool, reason: str, payload: dict | None = None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_actions(ts, mode, action_type, approved, reason, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (int(time.time()), str(mode), str(action_type), 1 if approved else 0, str(reason), json.dumps(payload or {})),
            )
            conn.commit()

    def log_tradingview_alert(
        self,
        ts: int,
        symbol: str | None,
        exchange: str | None,
        timeframe: str | None,
        signal: str | None,
        price: float | None,
        raw_json: str,
        idempotency_key: str,
        processed: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a TradingView alert to metrics.db.

        `extra` is accepted for forward compatibility (not yet stored).
        """
        _ = extra  # reserved
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tradingview_alerts(
                    ts, symbol, exchange, timeframe, signal, price, raw_json, idempotency_key, processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(ts),
                    (str(symbol).strip().upper() if symbol is not None else None),
                    (str(exchange).strip().upper() if exchange is not None else None),
                    (str(timeframe).strip() if timeframe is not None else None),
                    (str(signal).strip().upper() if signal is not None else None),
                    (float(price) if price is not None else None),
                    str(raw_json or "{}"),
                    str(idempotency_key or ""),
                    int(processed or 0),
                ),
            )
            conn.commit()


    def log_tradingview_backtest(
        self,
        symbol: str,
        timeframe: str = "",
        signal: str = "",
        bundle_path: str = "",
        best_strategy: str = "",
        best_pl: float | None = None,
        status: str = "ok",
        error: str = "",
    ) -> None:
        """
        Index a TradingView-triggered quick backtest run (PAPER-only).

        bundle_path is a JSON file created under logs/backtest/tv_autovalidation.
        """
        ts = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tradingview_backtests
                    (ts, symbol, timeframe, signal, bundle_path, best_strategy, best_pl, status, error)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol,
                    timeframe,
                    signal,
                    bundle_path,
                    best_strategy,
                    best_pl,
                    status,
                    error,
                ),
            )
            conn.commit()

    def latest_actions(self, limit: int = 20):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, mode, action_type, approved, reason FROM agent_actions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return rows

    def latest_tradingview_alerts(self, limit: int = 50, processed: Optional[int] = None):
        """Fetch recent TradingView alerts for UI/debug (optional)."""
        q = "SELECT ts, symbol, exchange, timeframe, signal, price, processed FROM tradingview_alerts"
        args: list[Any] = []
        if processed is not None:
            q += " WHERE processed = ?"
            args.append(int(processed))
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(q, tuple(args)).fetchall()
        return rows
