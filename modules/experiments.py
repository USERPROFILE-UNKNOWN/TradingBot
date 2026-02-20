"""Experiments and strategy registry persistence (experiments.db).

v5.19.0 additions:
- experiments
- strategy_registry
- deployments
- deployment_units
- config_snapshots
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional


class ExperimentsStore:
    def __init__(self, db_dir: str):
        self.path = os.path.join(db_dir, "experiments.db")
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
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    symbol TEXT,
                    regime TEXT,
                    hypothesis TEXT,
                    strategy_name TEXT,
                    score REAL,
                    status TEXT NOT NULL,
                    details_json TEXT,
                    config_hash TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_ts ON experiments(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_symbol ON experiments(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_name)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    strategy_version TEXT,
                    status TEXT NOT NULL,
                    params_json TEXT,
                    metadata_json TEXT,
                    created_ts INTEGER NOT NULL,
                    updated_ts INTEGER NOT NULL,
                    UNIQUE(strategy_name, strategy_version)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_registry_name ON strategy_registry(strategy_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_registry_status ON strategy_registry(status)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_version TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    metrics_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deployments_ts ON deployments(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deployments_strategy ON deployments(strategy_name)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_units (
                    deployment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    diff_text TEXT,
                    status TEXT NOT NULL,
                    approved_by TEXT,
                    rollback_pointer TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deployment_units_status ON deployment_units(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deployment_units_created_at ON deployment_units(created_at)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    snapshot_hash TEXT NOT NULL UNIQUE,
                    config_json TEXT NOT NULL,
                    source TEXT,
                    note TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cfgsnap_ts ON config_snapshots(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cfgsnap_hash ON config_snapshots(snapshot_hash)")

            conn.commit()

    def log_experiment(
        self,
        *,
        symbol: str = "",
        regime: str = "UNKNOWN",
        hypothesis: str = "",
        strategy_name: str = "",
        score: float = 0.0,
        status: str = "paper_candidate",
        details: Optional[Dict[str, Any]] = None,
        config_hash: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(
                    ts, symbol, regime, hypothesis, strategy_name, score, status, details_json, config_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    str(symbol or "").upper(),
                    str(regime or "UNKNOWN").upper(),
                    str(hypothesis or ""),
                    str(strategy_name or ""),
                    float(score or 0.0),
                    str(status or "paper_candidate"),
                    json.dumps(details or {}),
                    str(config_hash or ""),
                ),
            )
            conn.commit()

    def upsert_strategy(
        self,
        *,
        strategy_name: str,
        strategy_version: str = "v1",
        status: str = "paper_candidate",
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO strategy_registry(
                    strategy_name, strategy_version, status, params_json, metadata_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_name, strategy_version)
                DO UPDATE SET
                    status=excluded.status,
                    params_json=excluded.params_json,
                    metadata_json=excluded.metadata_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    str(strategy_name or ""),
                    str(strategy_version or "v1"),
                    str(status or "paper_candidate"),
                    json.dumps(params or {}),
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            conn.commit()

    def log_deployment(
        self,
        *,
        strategy_name: str,
        strategy_version: str = "v1",
        stage: str = "PAPER",
        status: str = "candidate",
        reason: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO deployments(ts, strategy_name, strategy_version, stage, status, reason, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    str(strategy_name or ""),
                    str(strategy_version or "v1"),
                    str(stage or "PAPER").upper(),
                    str(status or "candidate"),
                    str(reason or ""),
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()

    def create_deployment_unit(
        self,
        *,
        deployment_id: str,
        change_type: str,
        diff_text: str = "",
        status: str = "PROPOSED",
        approved_by: str = "",
        rollback_pointer: str = "",
        created_at: str = "",
    ) -> None:
        created = str(created_at or "") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO deployment_units(
                    deployment_id, created_at, change_type, diff_text, status, approved_by, rollback_pointer
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(deployment_id)
                DO UPDATE SET
                    change_type=excluded.change_type,
                    diff_text=excluded.diff_text,
                    status=excluded.status,
                    approved_by=excluded.approved_by,
                    rollback_pointer=excluded.rollback_pointer
                """,
                (
                    str(deployment_id or ""),
                    created,
                    str(change_type or "UNKNOWN"),
                    str(diff_text or ""),
                    str(status or "PROPOSED").upper(),
                    str(approved_by or ""),
                    str(rollback_pointer or ""),
                ),
            )
            conn.commit()

    def update_deployment_unit_status(
        self,
        *,
        deployment_id: str,
        status: str,
        approved_by: str = "",
        rollback_pointer: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE deployment_units
                SET status=?, approved_by=CASE WHEN ?='' THEN approved_by ELSE ? END,
                    rollback_pointer=CASE WHEN ?='' THEN rollback_pointer ELSE ? END
                WHERE deployment_id=?
                """,
                (
                    str(status or "").upper(),
                    str(approved_by or ""),
                    str(approved_by or ""),
                    str(rollback_pointer or ""),
                    str(rollback_pointer or ""),
                    str(deployment_id or ""),
                ),
            )
            conn.commit()

    def get_deployment_unit(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT deployment_id, created_at, change_type, diff_text, status, approved_by, rollback_pointer
                FROM deployment_units
                WHERE deployment_id=?
                LIMIT 1
                """,
                (str(deployment_id or ""),),
            )
            r = cur.fetchone()
            if not r:
                return None
            return {
                "deployment_id": r[0],
                "created_at": r[1],
                "change_type": r[2],
                "diff_text": r[3] or "",
                "status": r[4] or "",
                "approved_by": r[5] or "",
                "rollback_pointer": r[6] or "",
            }

    def upsert_config_snapshot(
        self,
        *,
        snapshot_hash: str,
        config_json: Dict[str, Any],
        source: str = "agent_master",
        note: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO config_snapshots(ts, snapshot_hash, config_json, source, note)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_hash)
                DO UPDATE SET
                    config_json=excluded.config_json,
                    source=excluded.source,
                    note=excluded.note
                """,
                (
                    int(time.time()),
                    str(snapshot_hash or ""),
                    json.dumps(config_json or {}),
                    str(source or "agent_master"),
                    str(note or ""),
                ),
            )
            conn.commit()
