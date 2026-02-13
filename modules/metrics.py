"""Metrics persistence for AI Agent health and actions."""

from __future__ import annotations

import os
import sqlite3
import time
import json


class MetricsStore:
    def __init__(self, db_dir: str):
        self.path = os.path.join(db_dir, "metrics.db")
        os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path, check_same_thread=False)

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

    def latest_actions(self, limit: int = 20):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, mode, action_type, approved, reason FROM agent_actions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return rows
