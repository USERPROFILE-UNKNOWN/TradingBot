from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import RepoBase


class CandidatesRepo(RepoBase):

    """Candidate scan persistence.

    Stored in decision_logs.db (candidates table).
    """

    def ensure_schema(self) -> None:
        """Create required tables/indexes if missing."""
        lock = self._lock("decision_logs")
        with lock:
            conn = self._conn("decision_logs")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    scan_id TEXT,
                    scan_ts TEXT,
                    universe TEXT,
                    policy TEXT,
                    symbol TEXT,
                    score REAL,
                    reason TEXT,
                    details_json TEXT,
                    signals_json TEXT,
                    PRIMARY KEY (scan_id, universe, symbol)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_scan_ts ON candidates(scan_ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON candidates(symbol)")
            conn.commit()



    def save_candidates(self, scan_id: str, rows: list, universe: str = "AUTO", policy: str = "AUTO", scan_ts: str = None) -> int:
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        if not rows:
            return 0
        scan_ts = scan_ts or datetime.now().isoformat(sep=' ', timespec='seconds')
        inserted = 0
        with lock:
            try:
                cur = conn.cursor()
                for r in rows:
                    try:
                        details_json = json.dumps(r.get('details', {}), ensure_ascii=False)
                    except Exception:
                        details_json = "{}"
                    try:
                        signals_json = json.dumps(r.get('signals', {}), ensure_ascii=False)
                    except Exception:
                        signals_json = "{}"
                    try:
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO candidates
                            (scan_id, scan_ts, universe, policy, symbol, score, reason, details_json, signals_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(scan_id),
                                scan_ts,
                                str((r or {}).get('universe') or universe or "AUTO"),
                                str(policy),
                                str(r.get('symbol', '')).upper(),
                                float(r.get('score', 0.0)),
                                str(r.get('reason', '')),
                                details_json,
                                signals_json,
                            ),
                        )
                        inserted += 1
                    except Exception:
                        continue
                conn.commit()
            except Exception:
                try:
                    if self._logger:
                        self._logger.exception("DB Save Candidates Error")
                except Exception:
                    pass
        return inserted

    def get_latest_candidates(self, limit: int = 200):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        """Return latest candidate rows (prefer TradingView universe)."""
        with lock:
            try:
                cur = conn.cursor()
                cur.execute("SELECT scan_id, MAX(scan_ts) FROM candidates")
                row = cur.fetchone()
                if not row or not row[0]:
                    return pd.DataFrame()
                sid = row[0]

                # TradingView latest
                q_tv = """
                    SELECT scan_id, scan_ts, universe, policy, symbol, score, reason, details_json, signals_json
                    FROM candidates
                    WHERE scan_id=? AND UPPER(universe)='TRADINGVIEW'
                    ORDER BY score DESC
                    LIMIT ?
                """
                df_tv = pd.read_sql_query(q_tv, conn, params=(sid, int(max(limit, 200))))

                # fallback non-TV latest
                q = """
                    SELECT scan_id, scan_ts, universe, policy, symbol, score, reason, details_json, signals_json
                    FROM candidates
                    WHERE scan_id=?
                    ORDER BY score DESC
                    LIMIT ?
                """
                df_latest = pd.read_sql_query(q, conn, params=(sid, int(max(limit, 200))))

            except Exception:
                return pd.DataFrame()

        frames = []
        if isinstance(df_tv, pd.DataFrame) and not df_tv.empty:
            frames.append(df_tv)
        if isinstance(df_latest, pd.DataFrame) and not df_latest.empty:
            frames.append(df_latest)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        try:
            df['universe'] = df.get('universe').fillna('')
            # Prefer TradingView rows on symbol collisions.
            df['_pri'] = (df['universe'].astype(str).str.upper() != 'TRADINGVIEW').astype(int)
            df = df.sort_values(by=['_pri', 'score', 'scan_ts'], ascending=[True, False, False], kind='mergesort')
            df = df.drop_duplicates(subset=['symbol'], keep='first')
            df = df.head(int(limit))
            df = df.drop(columns=['_pri'], errors='ignore')
        except Exception:
            pass
        return df
