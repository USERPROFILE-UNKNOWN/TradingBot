import sqlite3
import hashlib
import pandas as pd
import time
import threading
from datetime import datetime
from contextlib import contextmanager
import os
import re
import shutil
import json
from typing import Callable, Any, Dict, Iterable, List, Optional, Tuple

from .logging_utils import get_logger
from .persistence.db_runtime import DbRuntime
from .persistence.repositories import (
    TradesRepo,
    HistoryRepo,
    DecisionsRepo,
    BacktestRepo,
    CandidatesRepo,
    AgentRepo,
)


class DataManager:
    def __init__(self, db_path, config=None, paths=None):
        """Database manager.

        Forward-only policy (v5.14.0 Tier 3): split DB layout only.

        - Live reads/writes use per-table DBs under CONFIGURATION->db_dir.
        - The db_path parameter is kept for call-site compatibility; it is not used for live reads/writes.

        Args:
            db_path: Legacy/compat path (ignored in split-only mode).
            config:  ConfigParser loaded by load_split_config().
            paths:   Path map from get_paths() (optional; used to resolve relative db_dir).
        """
        self.db_path = db_path
        self.config = config
        self.paths = paths or {}

        # v5.12.3 updateA: optional log callback (TradingApp.log) for DB diagnostics.
        # Safe default: None (falls back to standard module logger).
        self._log_cb = None
        self._logger = get_logger(__name__)

        self.db_mode = 'split'
        self.split_mode = True

        # In split mode, db_dir tells us where to create/read the per-table DB files.
        self.db_dir = self._read_db_dir(db_path, config, self.paths)

        # Backward-compatible attributes (single-mode uses these; split-mode keeps them as aliases)
        self.lock = threading.RLock()
        self.conn = None

        # Split-mode connection map: key -> sqlite3.Connection
        self.conns = {}
        self.locks = {}
        self.db_paths = {}
        self.runtime = None

        if self.split_mode:
            os.makedirs(self.db_dir, exist_ok=True)
            self._validate_split_db_layout()
            # v5.16.0: runtime split (connections/locks/pragmas centralized)
            self.runtime = DbRuntime(
                self.db_dir,
                paths=self.paths,
                logger=self._logger,
                read_agent_mode=self._read_agent_mode,
                log_fn=lambda m: self._log(m),
            )
            self.runtime.init_split_connections()

            # Backward-compatible aliases
            self.db_paths = self.runtime.db_paths
            self.conns = self.runtime.conns
            self.locks = self.runtime.locks
            self.conn = self.conns.get("historical_prices")


        # v5.16.1: repositories (thin DB layer). DataManager remains the public façade for now.
        self.trades_repo = TradesRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )
        self.history_repo = HistoryRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )
        self.decisions_repo = DecisionsRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )
        self.backtest_repo = BacktestRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )
        self.candidates_repo = CandidatesRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )
        self.agent_repo = AgentRepo(
            self.runtime if self.split_mode else None,
            split_mode=self.split_mode,
            conn_fallback=getattr(self, "conn", None),
            lock_fallback=getattr(self, "db_lock", None),
            logger=self._logger,
            log_fn=self._log,
            read_agent_mode=self._read_agent_mode,
        )

        self.ensure_schema()

    def set_log_callback(self, cb):
        """Set a runtime log callback.

        Callback compatibility:
        - Preferred: cb(msg, **context)
        - Legacy:    cb(msg)
        """
        try:
            self._log_cb = cb
        except Exception:
            self._log_cb = None
        self._logger = get_logger(__name__)

    def _log(self, msg: str, **context) -> None:
        """Best-effort log sink for DB-layer diagnostics."""
        ctx = {"component": "db", "mode": self._read_agent_mode(), **(context or {})}

        try:
            if callable(self._log_cb):
                try:
                    self._log_cb(str(msg), **ctx)
                except TypeError:
                    # Backward-compatible with legacy single-arg callbacks.
                    self._log_cb(str(msg))
                return
        except Exception:
            pass

        try:
            self._logger.info(msg, extra=ctx)
        except Exception:
            pass


    def _resolve_cfg_path(self, raw: str, base: Optional[str]) -> str:
        """Resolve a config path string into a normalized absolute/relative path.

        Relative paths are resolved against *base* when provided.
        """
        raw = (raw or "").strip()
        if not raw:
            return ""
        try:
            raw = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            pass
        try:
            if os.path.isabs(raw):
                return os.path.normpath(raw)
        except Exception:
            pass
        try:
            b = base or os.getcwd()
            return os.path.normpath(os.path.join(b, raw))
        except Exception:
            return raw
    def _validate_split_db_layout(self) -> None:
        """Validate expected split DB filenames exist, and log clearly if any are missing.

        NOTE: sqlite3.connect() will create missing files automatically. This check is purely
        for observability so users understand what is happening.
        """
        if not self.split_mode:
            return

        required = [
            "historical_prices.db",
            "active_trades.db",
            "trade_history.db",
            "decision_logs.db",
            "backtest_results.db",
        ]

        missing: List[str] = []
        try:
            for fn in required:
                p = os.path.join(self.db_dir, fn)
                if not os.path.exists(p):
                    missing.append(fn)
        except Exception:
            try:
                self._logger.exception("[E_DB_SPLIT_CHECK_FAIL] split DB layout check failed", extra={"component": "db", "mode": self._read_agent_mode()})
            except Exception:
                pass
            return

        if missing:
            self._log(
                "[E_DB_SPLIT_MISSING] "
                f"db_mode=split; missing split DB files in db_dir='{self.db_dir}': {', '.join(missing)}. "
                "Empty DBs will be created automatically."
            )


    def _read_db_mode(self, config) -> str:
        """Return the supported DB mode.

        Forward-only policy (v5.14.0 Tier 3): split DB layout only.
        """
        return 'split'

    def _read_db_dir(self, db_path: str, config, paths: dict) -> str:
        """Resolve CONFIGURATION->db_dir.

        - Absolute paths: used as-is.
        - Relative paths: resolved relative to TradingBot root (paths['root']) when available.
        - Fallback: directory containing db_path, or paths['db_dir'] if present.
        """
        raw = ""
        try:
            raw = (config.get("CONFIGURATION", "db_dir", fallback="") or "").strip()
        except Exception:
            raw = ""

        if raw:
            raw = os.path.expandvars(os.path.expanduser(raw))
            if os.path.isabs(raw):
                return os.path.normpath(raw)
            base = (paths or {}).get("root") or os.path.dirname(db_path)
            return os.path.normpath(os.path.join(base, raw))

        if (paths or {}).get("db_dir"):
            return os.path.normpath(paths["db_dir"])

        try:
            return os.path.normpath(os.path.dirname(db_path))
        except Exception:
            return os.path.normpath("db")

    def _read_agent_mode(self) -> str:
        """Best-effort agent mode for structured log context."""
        try:
            raw = self.config.get("CONFIGURATION", "agent_mode", fallback="OFF")
        except Exception:
            raw = "OFF"
        return str(raw or "OFF").strip().upper() or "OFF"

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        try:
            if not self.config:
                return default
            raw = self.config.get("CONFIGURATION", key, fallback=str(default))
            s = str(raw).strip().lower()
            return s in ("1", "true", "yes", "y", "on")
        except Exception:
            return default

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Delegate connection hardening to DbRuntime (Phase 2.5)."""
        try:
            rt = getattr(self, "runtime", None)
            if rt is not None:
                rt.configure_connection(conn)
                return
        except Exception:
            pass

        # Fallback (should be rare): apply the same conservative defaults inline.
        try:
            conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA temp_store=MEMORY")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA wal_autocheckpoint=2000")
        except Exception:
            pass

    def _conn(self, key: str):
        """Return the sqlite3.Connection for a given logical key."""
        if self.split_mode and getattr(self, "runtime", None) is not None:
            return self.runtime.conn(key)
        return self.conn

    def _lock(self, key: str):
        """Return the threading.Lock for a given logical key."""
        if self.split_mode and getattr(self, "runtime", None) is not None:
            return self.runtime.lock(key)
        return self.lock

    @contextmanager
    def _multi_lock(self, keys):
        """Acquire multiple locks in a deterministic order to avoid deadlocks."""
        if self.split_mode and getattr(self, "runtime", None) is not None:
            with self.runtime.multi_lock(keys):
                yield
            return

        with self.lock:
            yield


    def ensure_schema(self) -> None:
        """Ensure all split-DB schemas exist using repository-owned DDL.

        v5.16.2+: database.py no longer owns monolithic CREATE TABLE blocks.
        """
        try:
            self.trades_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_TRADES]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

        try:
            self.history_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_HISTORY]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

        try:
            self.decisions_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_DECISIONS]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

        try:
            self.candidates_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_CANDIDATES]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

        try:
            self.agent_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_AGENT]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

        try:
            self.backtest_repo.ensure_schema()
        except Exception:
            try:
                self._logger.exception('[E_DB_SCHEMA_BACKTEST]', extra={'component': 'db', 'mode': self._read_agent_mode()})
            except Exception:
                pass

    def export_db_health_report(self, out_dir: Optional[str] = None) -> Optional[str]:
        """Write a JSON DB health report into logs and return the file path."""
        try:
            ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            logs_dir = out_dir or (self.paths or {}).get("logs") or os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            out_path = os.path.join(logs_dir, f"[DB HEALTH] [{ts}].json")

            def _inspect_db(path: str, expected: List[Tuple[str, Optional[str]]]):
                info: Dict[str, object] = {
                    "path": path,
                    "exists": bool(path and os.path.exists(path)),
                }
                if not path:
                    return info

                try:
                    info["size_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
                except Exception:
                    info["size_bytes"] = None

                if not os.path.exists(path):
                    return info

                try:
                    conn = sqlite3.connect(path, timeout=2, check_same_thread=False)
                    try:
                        self._configure_connection(conn)
                    except Exception:
                        pass
                    try:
                        qc = conn.execute("PRAGMA quick_check(1)").fetchone()
                        info["quick_check"] = qc[0] if qc else None
                    except Exception:
                        info["quick_check"] = None

                    try:
                        rows = conn.execute(
                            "SELECT name, type FROM sqlite_master "
                            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
                        ).fetchall()
                        info["objects"] = [{"name": r[0], "type": r[1]} for r in (rows or [])]
                    except Exception:
                        info["objects"] = []

                    # Expected table checks + row counts + newest timestamp (if column provided)
                    checks = []
                    for table, ts_col in (expected or []):
                        one = {"table": table}
                        try:
                            row = conn.execute(
                                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                                (table,),
                            ).fetchone()
                            one["exists"] = bool(row)
                            if row:
                                try:
                                    cnt = conn.execute(f"SELECT COUNT(1) FROM {table}").fetchone()
                                    one["rows"] = int(cnt[0]) if cnt and cnt[0] is not None else 0
                                except Exception:
                                    one["rows"] = None
                                if ts_col:
                                    try:
                                        mx = conn.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()
                                        one["newest"] = mx[0] if mx else None
                                    except Exception:
                                        one["newest"] = None
                        except Exception:
                            one["exists"] = None
                        checks.append(one)
                    info["tables"] = checks

                    try:
                        conn.close()
                    except Exception:
                        pass
                except Exception as e:
                    info["error"] = str(e)

                return info

            report: Dict[str, object] = {
                "generated_at": ts,
                "db_mode": self.db_mode,
                "db_dir": self.db_dir,
                "legacy_db_path": self.db_path,
                "dbs": {},
            }

            if self.split_mode:
                # Split DBs (per-table)
                report["dbs"]["historical_prices"] = _inspect_db(
                    self.db_paths.get("historical_prices"),
                    [("historical_prices", "timestamp")],
                )
                report["dbs"]["active_trades"] = _inspect_db(
                    self.db_paths.get("active_trades"),
                    [("active_trades", "timestamp")],
                )
                report["dbs"]["trade_history"] = _inspect_db(
                    self.db_paths.get("trade_history"),
                    [("trade_history", "exit_time")],
                )
                report["dbs"]["decision_logs"] = _inspect_db(
                    self.db_paths.get("decision_logs"),
                    [("decision_logs", "timestamp")],
                )
                report["dbs"]["backtest_results"] = _inspect_db(
                    self.db_paths.get("backtest_results"),
                    [("backtest_results", "timestamp")],
                )
            else:
                # Legacy single DB
                report["dbs"]["market_data"] = _inspect_db(
                    self.db_path,
                    [
                        ("historical_prices", "timestamp"),
                        ("active_trades", "timestamp"),
                        ("trade_history", "exit_time"),
                        ("decision_logs", "timestamp"),
                        ("backtest_results", "timestamp"),
                    ],
                )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            return out_path
        except Exception:
            return None

    def repair_backtest_results_db(self, *, recreate_view: bool = True) -> Tuple[bool, str]:
        """Attempt to repair backtest_results.db (split mode) if schema becomes malformed.

        Returns:
            (ok, message)
        """
        if not self.split_mode:
            return False, "Repair is available only in split-db mode."

        path = (self.db_paths or {}).get("backtest_results") or os.path.join(self.db_dir, "backtest_results.db")
        ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        backup_root = (self.paths or {}).get("backup") or os.path.join(os.path.dirname(self.db_dir), "backups")
        bdir = os.path.join(backup_root, "db_repairs", ts)

        try:
            os.makedirs(bdir, exist_ok=True)
        except Exception:
            pass

        # Close existing connection if present
        try:
            c0 = (self.conns or {}).get("backtest_results")
            if c0:
                try:
                    c0.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Backup current file (if exists)
        try:
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(bdir, os.path.basename(path)))
        except Exception:
            pass

        recreated = False

        # Try to open and fix view; if schema is malformed, recreate DB file
        try:
            conn = sqlite3.connect(path, timeout=2, check_same_thread=False)
            self._configure_connection(conn)
            if recreate_view:
                try:
                    conn.execute("DROP VIEW IF EXISTS backtest_testing_data")
                except Exception:
                    pass
                try:
                    row = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='backtest_results' LIMIT 1"
                    ).fetchone()
                    if row:
                        conn.execute("CREATE VIEW IF NOT EXISTS backtest_testing_data AS SELECT * FROM backtest_results")
                except Exception:
                    pass
            try:
                conn.commit()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        except sqlite3.DatabaseError:
            # Malformed schema: delete and recreate
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            try:
                conn = sqlite3.connect(path, timeout=2, check_same_thread=False)
                self._configure_connection(conn)
                conn.commit()
                conn.close()
                recreated = True
            except Exception as e:
                return False, f"Failed to recreate backtest_results.db: {e}"

        # Re-open and re-attach to manager
        try:
            conn2 = sqlite3.connect(path, check_same_thread=False)
            self._configure_connection(conn2)
            self.conns["backtest_results"] = conn2
            self.locks["backtest_results"] = threading.RLock()
        except Exception as e:
            return False, f"Repair completed but could not reopen connection: {e}"

        if recreated:
            return True, f"Recreated backtest_results.db (backup saved to {bdir})."
        return True, f"Repaired backtest_results.db (backup saved to {bdir})."

    def log_decision(self, *args, **kwargs):
        return self.decisions_repo.log_decision(*args, **kwargs)

    def log_decision_packet(self, *args, **kwargs):
        return self.decisions_repo.log_decision_packet(*args, **kwargs)

    def log_execution_packet(self, *args, **kwargs):
        return self.decisions_repo.log_execution_packet(*args, **kwargs)

    def get_decision_packets(self, *args, **kwargs):
        return self.decisions_repo.get_decision_packets(*args, **kwargs)

    def get_execution_packets(self, *args, **kwargs):
        return self.decisions_repo.get_execution_packets(*args, **kwargs)

    def log_config_change(self, *args, **kwargs):
        return self.decisions_repo.log_config_change(*args, **kwargs)

    def log_watchlist_audit(self, *args, **kwargs):
        return self.decisions_repo.log_watchlist_audit(*args, **kwargs)

    def get_config_history(self, *args, **kwargs):
        return self.decisions_repo.get_config_history(*args, **kwargs)

    def get_recent_decisions(self, *args, **kwargs):
        return self.decisions_repo.get_recent_decisions(*args, **kwargs)

    def save_candidates(self, *args, **kwargs):
        return self.candidates_repo.save_candidates(*args, **kwargs)

    def get_latest_candidates(self, *args, **kwargs):
        return self.candidates_repo.get_latest_candidates(*args, **kwargs)

    def get_latest_snapshot(self, *args, **kwargs):
        return self.history_repo.get_latest_snapshot(*args, **kwargs)

    def rebuild_backtest_table(self, *args, **kwargs):
        return self.backtest_repo.rebuild_backtest_table(*args, **kwargs)

    def ensure_backtest_table(self, *args, **kwargs):
        return self.backtest_repo.ensure_backtest_table(*args, **kwargs)

    def save_backtest_result(self, *args, **kwargs):
        return self.backtest_repo.save_backtest_result(*args, **kwargs)

    def get_backtest_data(self, *args, **kwargs):
        return self.backtest_repo.get_backtest_data(*args, **kwargs)

    def get_best_strategy_for_symbol(self, *args, **kwargs):
        return self.backtest_repo.get_best_strategy_for_symbol(*args, **kwargs)

    def get_all_symbols(self, *args, **kwargs):
        return self.history_repo.get_all_symbols(*args, **kwargs)

    def save_bulk_data(self, *args, **kwargs):
        return self.history_repo.save_bulk_data(*args, **kwargs)

    def save_snapshot(self, *args, **kwargs):
        return self.history_repo.save_snapshot(*args, **kwargs)

    def get_last_timestamp(self, *args, **kwargs):
        return self.history_repo.get_last_timestamp(*args, **kwargs)

    def get_first_timestamp(self, *args, **kwargs):
        return self.history_repo.get_first_timestamp(*args, **kwargs)

    def get_history(self, *args, **kwargs):
        return self.history_repo.get_history(*args, **kwargs)

    def get_trade_markers(self, *args, **kwargs):
        return self.trades_repo.get_trade_markers(*args, **kwargs)

    def log_trade_entry(self, *args, **kwargs):
        return self.trades_repo.log_trade_entry(*args, **kwargs)

    def update_highest_price(self, *args, **kwargs):
        return self.trades_repo.update_highest_price(*args, **kwargs)

    def get_active_trades_rows(self, *args, **kwargs):
        return self.trades_repo.get_active_trades_rows(*args, **kwargs)

    def get_active_trades(self, *args, **kwargs):
        return self.trades_repo.get_active_trades(*args, **kwargs)

    def remove_active_trade(self, *args, **kwargs):
        return self.trades_repo.remove_active_trade(*args, **kwargs)

    def close_trade(self, *args, **kwargs):
        return self.trades_repo.close_trade(*args, **kwargs)

    def get_portfolio_stats(self, *args, **kwargs):
        return self.trades_repo.get_portfolio_stats(*args, **kwargs)

    def get_strategy_stats(self, *args, **kwargs):
        return self.trades_repo.get_strategy_stats(*args, **kwargs)

    def get_recent_history(self, *args, **kwargs):
        return self.trades_repo.get_recent_history(*args, **kwargs)

    def get_trade_history_since(self, *args, **kwargs):
        return self.trades_repo.get_trade_history_since(*args, **kwargs)

    def get_decision_counts_since(self, *args, **kwargs):
        return self.decisions_repo.get_decision_counts_since(*args, **kwargs)

    def get_latest_timestamps_for_symbols(self, *args, **kwargs):
        return self.history_repo.get_latest_timestamps_for_symbols(*args, **kwargs)

    def upsert_agent_suggestion(self, *args, **kwargs):
        return self.agent_repo.upsert_agent_suggestion(*args, **kwargs)

    def add_agent_rationale(self, *args, **kwargs):
        return self.agent_repo.add_agent_rationale(*args, **kwargs)

    def get_agent_suggestions(self, *args, **kwargs):
        return self.agent_repo.get_agent_suggestions(*args, **kwargs)

    def get_agent_suggestion_detail(self, *args, **kwargs):
        return self.agent_repo.get_agent_suggestion_detail(*args, **kwargs)

    def set_agent_suggestion_status(self, *args, **kwargs):
        return self.agent_repo.set_agent_suggestion_status(*args, **kwargs)

