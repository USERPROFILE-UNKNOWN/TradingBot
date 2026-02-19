"""Metrics and audit persistence (metrics.db).

v5.17.0 additions:
- job_runs / anomalies / symbol_health tables
- health snapshot query helper for Dashboard widget
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms INTEGER,
                    details_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_runs_ts ON job_runs(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_runs_name ON job_runs(job_name)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_run_state (
                    job_name TEXT PRIMARY KEY,
                    last_success_at INTEGER,
                    last_attempt_at INTEGER,
                    cooldown_until INTEGER,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autopilot_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at INTEGER NOT NULL,
                    ended_at INTEGER,
                    mode TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_autopilot_runs_started_at ON autopilot_runs(started_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT,
                    details_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON anomalies(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(event_type)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    freshness_seconds REAL,
                    api_error_streak INTEGER,
                    reject_ratio REAL,
                    decision_exec_latency_ms REAL,
                    slippage_bps REAL,
                    details_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_health_ts ON symbol_health(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_health_symbol ON symbol_health(symbol)")

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

    def log_job_run(
        self,
        job_name: str,
        status: str,
        duration_ms: int | None = None,
        details: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO job_runs(ts, job_name, status, duration_ms, details_json) VALUES (?, ?, ?, ?, ?)",
                (
                    int(time.time()),
                    str(job_name),
                    str(status),
                    (int(duration_ms) if duration_ms is not None else None),
                    json.dumps(details or {}),
                ),
            )
            conn.commit()

    def get_job_state(self, job_name: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_success_at, last_attempt_at, cooldown_until, last_error FROM job_run_state WHERE job_name = ?",
                (str(job_name),),
            ).fetchone()
        if not row:
            return {}
        return {
            "last_success_at": (int(row[0]) if row[0] is not None else None),
            "last_attempt_at": (int(row[1]) if row[1] is not None else None),
            "cooldown_until": (int(row[2]) if row[2] is not None else None),
            "last_error": (str(row[3]) if row[3] is not None else ""),
        }

    def update_job_state(self, job_name: str, status: str, *, error: str = "", cooldown_until: int | None = None) -> None:
        now_ts = int(time.time())
        ok = str(status).strip().lower() == "ok"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_run_state(job_name, last_success_at, last_attempt_at, cooldown_until, last_error)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_name) DO UPDATE SET
                    last_success_at=excluded.last_success_at,
                    last_attempt_at=excluded.last_attempt_at,
                    cooldown_until=excluded.cooldown_until,
                    last_error=excluded.last_error
                """,
                (
                    str(job_name),
                    (now_ts if ok else None),
                    now_ts,
                    (int(cooldown_until) if cooldown_until is not None else None),
                    ("" if ok else str(error or "")),
                ),
            )
            conn.commit()

    def start_autopilot_run(self, run_id: str, mode: str, phase: str, status: str = "OK", summary: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO autopilot_runs(run_id, started_at, ended_at, mode, phase, status, summary_json)
                VALUES (?, ?, NULL, ?, ?, ?, ?)
                """,
                (str(run_id), int(time.time()), str(mode), str(phase), str(status), json.dumps(summary or {})),
            )
            conn.commit()

    def finish_autopilot_run(self, run_id: str, status: str = "OK", summary: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE autopilot_runs
                   SET ended_at = ?, status = ?, summary_json = ?
                 WHERE run_id = ?
                """,
                (int(time.time()), str(status), json.dumps(summary or {}), str(run_id)),
            )
            conn.commit()

    def log_anomaly(
        self,
        event_type: str,
        severity: str = "WARN",
        source: str = "agent_master",
        details: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO anomalies(ts, event_type, severity, source, details_json) VALUES (?, ?, ?, ?, ?)",
                (int(time.time()), str(event_type), str(severity), str(source), json.dumps(details or {})),
            )
            conn.commit()

    def log_symbol_health(
        self,
        symbol: str,
        *,
        freshness_seconds: float | None = None,
        api_error_streak: int | None = None,
        reject_ratio: float | None = None,
        decision_exec_latency_ms: float | None = None,
        slippage_bps: float | None = None,
        details: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_health(
                    ts, symbol, freshness_seconds, api_error_streak, reject_ratio,
                    decision_exec_latency_ms, slippage_bps, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    str(symbol).upper(),
                    (float(freshness_seconds) if freshness_seconds is not None else None),
                    (int(api_error_streak) if api_error_streak is not None else None),
                    (float(reject_ratio) if reject_ratio is not None else None),
                    (float(decision_exec_latency_ms) if decision_exec_latency_ms is not None else None),
                    (float(slippage_bps) if slippage_bps is not None else None),
                    json.dumps(details or {}),
                ),
            )
            conn.commit()

    def get_health_widget_snapshot(self, lookback_sec: int = 3600) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "data_freshness_seconds": None,
            "api_error_streak": 0,
            "order_reject_ratio": 0.0,
            "decision_exec_latency_ms": None,
            "slippage_bps": None,
            "anomaly_count": 0,
        }
        since_ts = int(time.time()) - max(60, int(lookback_sec))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT freshness_seconds, api_error_streak, reject_ratio, decision_exec_latency_ms, slippage_bps "
                "FROM symbol_health WHERE ts >= ? ORDER BY id DESC LIMIT 1",
                (since_ts,),
            ).fetchone()
            if row:
                out["data_freshness_seconds"] = row[0]
                out["api_error_streak"] = int(row[1] or 0)
                out["order_reject_ratio"] = float(row[2] or 0.0)
                out["decision_exec_latency_ms"] = row[3]
                out["slippage_bps"] = row[4]
            arow = conn.execute("SELECT COUNT(1) FROM anomalies WHERE ts >= ?", (since_ts,)).fetchone()
            out["anomaly_count"] = int((arow or [0])[0] or 0)
        return out

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
