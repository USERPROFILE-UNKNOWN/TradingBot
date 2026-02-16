from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd

from .base import RepoBase


class AgentRepo(RepoBase):

    """Agent persistence: suggestions, rationales, and audit trail.

    Stored in decision_logs.db (agent_suggestions / agent_rationales).
    """


    def ensure_schema(self) -> None:
        """Create required tables/indexes if missing."""
        lock = self._lock("decision_logs")
        with lock:
            conn = self._conn("decision_logs")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    artifact_type TEXT,
                    artifact_path TEXT,
                    title TEXT,
                    suggestion_type TEXT,
                    suggestion_json TEXT,
                    fingerprint TEXT UNIQUE,
                    status TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_suggestions_created_at ON agent_suggestions(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_suggestions_status ON agent_suggestions(status)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_rationales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_id INTEGER,
                    created_at TEXT,
                    severity TEXT,
                    rationale_text TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_rationales_suggestion_id ON agent_rationales(suggestion_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_rationales_created_at ON agent_rationales(created_at)")
            conn.commit()


    def _agent_fingerprint(self, title: str, suggestion_type: str, suggestion_json: str, artifact_path: str) -> str:
        '''Stable hash for deduping suggestions.'''
        raw = f"{title}|{suggestion_type}|{artifact_path}|{suggestion_json}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def upsert_agent_suggestion(
        self,
        artifact_type: str,
        artifact_path: str,
        title: str,
        suggestion_type: str,
        suggestion_payload: dict,
        status: str = "NEW",
    ):
        '''Insert a new suggestion if fingerprint doesn't exist; returns suggestion_id.'''
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        suggestion_json = json.dumps(suggestion_payload, ensure_ascii=False)
        fp = self._agent_fingerprint(title, suggestion_type, suggestion_json, artifact_path)

        with lock:
            cur = conn.cursor()
            cur.execute("SELECT id FROM agent_suggestions WHERE fingerprint = ?", (fp,))
            row = cur.fetchone()
            if row:
                return int(row[0])

            cur.execute(
                "INSERT INTO agent_suggestions (created_at, artifact_type, artifact_path, title, suggestion_type, suggestion_json, fingerprint, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (created_at, artifact_type, artifact_path, title, suggestion_type, suggestion_json, fp, status),
            )
            conn.commit()
            return int(cur.lastrowid)

    def add_agent_rationale(self, suggestion_id: int, rationale_text: str, severity: str = "INFO"):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with lock:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO agent_rationales (suggestion_id, created_at, severity, rationale_text) VALUES (?, ?, ?, ?)",
                (int(suggestion_id), created_at, severity, rationale_text),
            )
            conn.commit()

    def get_agent_suggestions(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT id, created_at, artifact_type, artifact_path, title, suggestion_type, suggestion_json, status "
                    "FROM agent_suggestions ORDER BY id DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
            except Exception:
                return None
        if df is None or df.empty:
            return df
        return df

    def get_agent_suggestion_detail(self, suggestion_id: int):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT id, created_at, artifact_type, artifact_path, title, suggestion_type, suggestion_json, status "
                    "FROM agent_suggestions WHERE id = ?",
                    conn,
                    params=(int(suggestion_id),),
                )
                df2 = pd.read_sql_query(
                    "SELECT created_at, severity, rationale_text FROM agent_rationales WHERE suggestion_id = ? ORDER BY id ASC",
                    conn,
                    params=(int(suggestion_id),),
                )
            except Exception:
                return None, None
        return df, df2

    def set_agent_suggestion_status(self, suggestion_id: int, status: str):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            cur = conn.cursor()
            cur.execute(
                "UPDATE agent_suggestions SET status = ? WHERE id = ?",
                (str(status), int(suggestion_id)),
            )
            conn.commit()
