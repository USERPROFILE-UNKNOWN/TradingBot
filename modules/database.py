import sqlite3
import hashlib
import pandas as pd
import time
import threading
from datetime import datetime
from contextlib import ExitStack, contextmanager
import os
import re
import shutil
import json
from typing import Callable, Any, Dict, Iterable, List, Optional, Tuple

from .logging_utils import get_logger


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
        # Safe default: None (falls back to print in a few places).
        self._log_cb = None
        self._logger = get_logger(__name__)

        self.db_mode = 'split'
        self.split_mode = True

        # In split mode, db_dir tells us where to create/read the per-table DB files.
        self.db_dir = self._read_db_dir(db_path, config, self.paths)

        # Backward-compatible attributes (single-mode uses these; split-mode keeps them as aliases)
        self.lock = threading.Lock()
        self.conn = None

        # Split-mode connection map: key -> sqlite3.Connection
        self.conns = {}
        self.locks = {}
        self.db_paths = {}

        if self.split_mode:
            os.makedirs(self.db_dir, exist_ok=True)
            self._validate_split_db_layout()
            self._init_split_connections()

        self.create_tables()

        # v5.12.0 Update D: DB hardening (safe indexes; health/repair tools use the same helpers)
        try:
            self.ensure_db_indexes()
        except Exception:
            pass

    def set_log_callback(self, cb):
        """Set a callable that accepts a single string for runtime logging."""
        try:
            self._log_cb = cb
        except Exception:
            self._log_cb = None
        self._logger = get_logger(__name__)

    def _log(self, msg: str) -> None:
        """Best-effort log sink for DB-layer diagnostics."""
        try:
            if callable(self._log_cb):
                self._log_cb(str(msg))
                return
        except Exception:
            pass
        try:
            self._logger.info(msg)
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
            return

        if missing:
            self._log(
                "[DB] E_DB_SPLIT_MISSING | "
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
        - Relative paths: resolved relative to TradingBot/config/ (paths['config_dir']) when available.
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
            base = (paths or {}).get("config_dir") or os.path.dirname(db_path)
            return os.path.normpath(os.path.join(base, raw))

        if (paths or {}).get("db_dir"):
            return os.path.normpath(paths["db_dir"])

        try:
            return os.path.normpath(os.path.dirname(db_path))
        except Exception:
            return os.path.normpath("db")

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
        """Apply conservative PRAGMA settings for better concurrency + stability.

        Notes:
          - WAL improves read/write concurrency (especially updater writes + UI reads).
          - synchronous=NORMAL is the recommended pairing with WAL for performance while remaining safe.
          - busy_timeout reduces transient 'database is locked' errors.
        """
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
    def _init_split_connections(self) -> None:
        """Open one SQLite connection per logical table group."""
        self.db_paths = {
            "historical_prices": os.path.join(self.db_dir, "historical_prices.db"),
            "active_trades": os.path.join(self.db_dir, "active_trades.db"),
            "trade_history": os.path.join(self.db_dir, "trade_history.db"),
            "decision_logs": os.path.join(self.db_dir, "decision_logs.db"),
            "backtest_results": os.path.join(self.db_dir, "backtest_results.db"),
        }
        for key, path in self.db_paths.items():
            try:
                conn = sqlite3.connect(path, check_same_thread=False)
                self._configure_connection(conn)
                # Force schema load early so we can detect malformed schema (e.g. bad views)
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            except sqlite3.DatabaseError as e:
                # Known issue (v5.10.0 migration): a view was created with a schema qualifier
                # (e.g. bt.backtest_results). That can render the DB schema 'malformed'.
                msg = str(e)
                if key == "backtest_results" and (
                    "malformed database schema" in msg
                    or "cannot reference objects in database" in msg
                    or "backtest_testing_data" in msg
                ):
                    try:
                        # Backup the corrupted DB and recreate a clean one.
                        ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
                        backup_root = (self.paths or {}).get("backup") or os.path.join(os.path.dirname(self.db_dir), "backups")
                        bdir = os.path.join(backup_root, "db_repairs", ts)
                        os.makedirs(bdir, exist_ok=True)
                        shutil.copy2(path, os.path.join(bdir, os.path.basename(path)))
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        self._logger.warning("[DB] ⚠️ backtest_results.db schema was malformed; backed up to: %s", bdir)
                    except Exception:
                        pass
                    conn = sqlite3.connect(path, check_same_thread=False)
                    self._configure_connection(conn)
                else:
                    # For other DBs, re-raise (should not happen in normal operation)
                    raise

            self.conns[key] = conn
            self.locks[key] = threading.Lock()

        # Convenience alias: treat "conn" as the primary read-heavy DB
        self.conn = self.conns["historical_prices"]

    def _conn(self, key: str):
        """Return the sqlite3.Connection for a given logical key."""
        if self.split_mode:
            return self.conns[key]
        return self.conn

    def _lock(self, key: str):
        """Return the threading.Lock for a given logical key."""
        if self.split_mode:
            return self.locks.get(key, self.lock)
        return self.lock

    @contextmanager
    def _multi_lock(self, keys):
        """Acquire multiple locks in a deterministic order to avoid deadlocks."""
        if not self.split_mode:
            with self.lock:
                yield
            return

        uniq = sorted(set(keys))
        with ExitStack() as stack:
            for k in uniq:
                lk = self.locks.get(k)
                if lk is None:
                    lk = threading.Lock()
                    self.locks[k] = lk
                stack.enter_context(lk)
            yield

    def create_tables(self):
        """Create required tables.

        In 'split' mode, each table is created inside its own DB file.
        In 'single' mode, the legacy market_data.db contains all tables.
        """
        if self.split_mode:
            # historical_prices.db
            conn = self._conn("historical_prices")
            lock = self._lock("historical_prices")
            with lock:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS historical_prices (
                        symbol TEXT, timestamp DATETIME, close REAL, open REAL, high REAL, low REAL,
                        volume REAL, rsi REAL, bb_lower REAL, bb_upper REAL, ema_200 REAL, adx REAL, atr REAL,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                conn.commit()

            # active_trades.db
            conn = self._conn("active_trades")
            lock = self._lock("active_trades")
            with lock:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS active_trades (
                        symbol TEXT PRIMARY KEY,
                        qty INTEGER,
                        entry_price REAL,
                        highest_price REAL,
                        strategy TEXT,
                        timestamp DATETIME
                    )
                ''')
                conn.commit()

            # trade_history.db
            conn = self._conn("trade_history")
            lock = self._lock("trade_history")
            with lock:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        qty INTEGER,
                        entry_price REAL,
                        exit_price REAL,
                        profit_loss REAL,
                        strategy TEXT,
                        entry_time DATETIME,
                        exit_time DATETIME
                    )
                ''')
                conn.commit()

            # decision_logs.db
            conn = self._conn("decision_logs")
            lock = self._lock("decision_logs")
            with lock:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS decision_logs (
                        timestamp DATETIME,
                        symbol TEXT,
                        strategy TEXT,
                        action TEXT,
                        price REAL,
                        rsi REAL,
                        ai_score REAL,
                        sentiment REAL,
                        reason TEXT
                    )
                ''')
                conn.commit()


                # v5.12.6 updateA: decision/execution packets (replay harness)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS decision_packets (
                        decision_id TEXT PRIMARY KEY,
                        timestamp DATETIME,
                        symbol TEXT,
                        strategy TEXT,
                        action TEXT,
                        score REAL,
                        price REAL,
                        ai_prob REAL,
                        sentiment REAL,
                        reason TEXT,
                        market_regime TEXT,
                        is_crypto INTEGER,
                        payload_json TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS execution_packets (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME,
                        decision_id TEXT,
                        symbol TEXT,
                        side TEXT,
                        phase TEXT,
                        qty REAL,
                        price REAL,
                        order_id TEXT,
                        client_order_id TEXT,
                        broker_status TEXT,
                        payload_json TEXT
                    )
                ''')
                # indexes (safe)
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_decision_packets_symbol_ts ON decision_packets(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_decision_ts ON execution_packets(decision_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_order_id ON execution_packets(order_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_client_order_id ON execution_packets(client_order_id)')

                # v5.12.7 updateA: configuration change history (Config tab foundation)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS config_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        cfg_key TEXT,
                        old_value TEXT,
                        new_value TEXT,
                        source TEXT
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_history_ts ON config_history(timestamp)')

                # Agent Shadow Mode (v5.13.0 Update A)
                cursor.execute('CREATE TABLE IF NOT EXISTS agent_suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at DATETIME, artifact_type TEXT, artifact_path TEXT, title TEXT, suggestion_type TEXT, suggestion_json TEXT, fingerprint TEXT UNIQUE, status TEXT, applied_at DATETIME, applied_by TEXT)')
                cursor.execute('CREATE TABLE IF NOT EXISTS agent_rationales (id INTEGER PRIMARY KEY AUTOINCREMENT, suggestion_id INTEGER, created_at DATETIME, rationale TEXT, metrics_json TEXT, FOREIGN KEY(suggestion_id) REFERENCES agent_suggestions(id))')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_suggestions_status_ts ON agent_suggestions(status, created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_rationales_suggestion_id ON agent_rationales(suggestion_id)')
                # v5.13.1 updateA: Candidate scanner results
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id TEXT,
                        scan_ts DATETIME,
                        scan_date TEXT,
                        symbol TEXT,
                        score REAL,
                        ret_lookback REAL,
                        dollar_volume REAL,
                        volatility REAL,
                        bars INTEGER,
                        universe TEXT,
                        details_json TEXT,
                        UNIQUE(scan_id, symbol)
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_candidates_scan_date_score ON candidates(scan_date, score)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_candidates_symbol_ts ON candidates(symbol, scan_ts)')

                # v5.13.1 updateB: Watchlist policy audit log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS watchlist_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT,
                        timestamp DATETIME,
                        source TEXT,
                        mode TEXT,
                        before_json TEXT,
                        after_json TEXT,
                        added_json TEXT,
                        removed_json TEXT,
                        meta_json TEXT
                    )
                ''')
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_audit_batch ON watchlist_audit(batch_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_audit_ts ON watchlist_audit(timestamp)')


                cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_history_key_ts ON config_history(cfg_key, timestamp)')
                conn.commit()

            # backtest_results.db is created/rebuilt on-demand via rebuild_backtest_table()
            return

        # Legacy single-db mode
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_prices (
                    symbol TEXT, timestamp DATETIME, close REAL, open REAL, high REAL, low REAL,
                    volume REAL, rsi REAL, bb_lower REAL, bb_upper REAL, ema_200 REAL, adx REAL, atr REAL,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_trades (
                    symbol TEXT PRIMARY KEY,
                    qty INTEGER,
                    entry_price REAL,
                    highest_price REAL,
                    strategy TEXT,
                    timestamp DATETIME
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    qty INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    strategy TEXT,
                    entry_time DATETIME,
                    exit_time DATETIME
                )
            ''')
            # v3.3: Black Box Decision Log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_logs (
                    timestamp DATETIME,
                    symbol TEXT,
                    strategy TEXT,
                    action TEXT,
                    price REAL,
                    rsi REAL,
                    ai_score REAL,
                    sentiment REAL,
                    reason TEXT
                )
            ''')
            self.conn.commit()

            # v5.12.6 updateA: decision/execution packets (replay harness)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_packets (
                    decision_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    strategy TEXT,
                    action TEXT,
                    score REAL,
                    price REAL,
                    ai_prob REAL,
                    sentiment REAL,
                    reason TEXT,
                    market_regime TEXT,
                    is_crypto INTEGER,
                    payload_json TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_packets (
                    event_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    decision_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    phase TEXT,
                    qty REAL,
                    price REAL,
                    order_id TEXT,
                    client_order_id TEXT,
                    broker_status TEXT,
                    payload_json TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decision_packets_symbol_ts ON decision_packets(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_decision_ts ON execution_packets(decision_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_order_id ON execution_packets(order_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_packets_client_order_id ON execution_packets(client_order_id)')

            # v5.12.7 updateA: configuration change history (Config tab foundation)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    cfg_key TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    source TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_history_ts ON config_history(timestamp)')

            # Agent Shadow Mode (v5.13.0 Update A)
            cursor.execute('CREATE TABLE IF NOT EXISTS agent_suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at DATETIME, artifact_type TEXT, artifact_path TEXT, title TEXT, suggestion_type TEXT, suggestion_json TEXT, fingerprint TEXT UNIQUE, status TEXT, applied_at DATETIME, applied_by TEXT)')
            cursor.execute('CREATE TABLE IF NOT EXISTS agent_rationales (id INTEGER PRIMARY KEY AUTOINCREMENT, suggestion_id INTEGER, created_at DATETIME, rationale TEXT, metrics_json TEXT, FOREIGN KEY(suggestion_id) REFERENCES agent_suggestions(id))')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_suggestions_status_ts ON agent_suggestions(status, created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_rationales_suggestion_id ON agent_rationales(suggestion_id)')
            # v5.13.1 updateA: Candidate scanner results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT,
                    scan_ts DATETIME,
                    scan_date TEXT,
                    symbol TEXT,
                    score REAL,
                    ret_lookback REAL,
                    dollar_volume REAL,
                    volatility REAL,
                    bars INTEGER,
                    universe TEXT,
                    details_json TEXT,
                    UNIQUE(scan_id, symbol)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_candidates_scan_date_score ON candidates(scan_date, score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_candidates_symbol_ts ON candidates(symbol, scan_ts)')

            # v5.13.1 updateB: Watchlist policy audit log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    timestamp DATETIME,
                    source TEXT,
                    mode TEXT,
                    before_json TEXT,
                    after_json TEXT,
                    added_json TEXT,
                    removed_json TEXT,
                    meta_json TEXT
                )
            ''')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_audit_batch ON watchlist_audit(batch_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_audit_ts ON watchlist_audit(timestamp)')


            cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_history_key_ts ON config_history(cfg_key, timestamp)')
            self.conn.commit()


    
    # --------------------
    # v5.12.0 Update D: DB indexes + health/repair helpers
    # --------------------
    def ensure_db_indexes(self):
        """Create lightweight indexes (idempotent, safe).

        Uses IF NOT EXISTS and skips missing tables/columns.
        """

        def _table_exists(conn, table: str) -> bool:
            try:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                    (table,),
                ).fetchone()
                return bool(row)
            except Exception:
                return False

        def _cols(conn, table: str) -> set:
            try:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                return {r[1] for r in (rows or [])}
            except Exception:
                return set()

        def _safe(conn, sql: str) -> None:
            try:
                conn.execute(sql)
                conn.commit()
            except Exception:
                pass

        # decision_logs indexes
        try:
            c = self._conn("decision_logs")
            if _table_exists(c, "decision_logs"):
                cols = _cols(c, "decision_logs")
                if "timestamp" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_decision_logs_timestamp ON decision_logs(timestamp)")
                if "symbol" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol ON decision_logs(symbol)")
        except Exception:
            pass

        # trade_history indexes
        try:
            c = self._conn("trade_history")
            if _table_exists(c, "trade_history"):
                cols = _cols(c, "trade_history")
                if "exit_time" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_trade_history_exit_time ON trade_history(exit_time)")
                if "entry_time" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_trade_history_entry_time ON trade_history(entry_time)")
                if "symbol" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)")
        except Exception:
            pass

        # backtest_results index (when table exists)
        try:
            c = self._conn("backtest_results")
            if _table_exists(c, "backtest_results"):
                cols = _cols(c, "backtest_results")
                if "timestamp" in cols:
                    _safe(c, "CREATE INDEX IF NOT EXISTS idx_backtest_results_timestamp ON backtest_results(timestamp)")
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
            self.locks["backtest_results"] = threading.Lock()
        except Exception as e:
            return False, f"Repair completed but could not reopen connection: {e}"

        if recreated:
            return True, f"Recreated backtest_results.db (backup saved to {bdir})."
        return True, f"Repaired backtest_results.db (backup saved to {bdir})."

    def log_decision(self, symbol, strategy, action, price, rsi, ai_score, sentiment, reason):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO decision_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now(), symbol, strategy, action, price, rsi, ai_score, sentiment, reason))
                conn.commit()
            except Exception:
                self._logger.exception("Log Decision Error")


    # --------------------
    # v5.12.6 updateA: Replay Harness (Decision/Execution Packets)
    # --------------------
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
                        float(score) if score is not None else 0.0,
                        float(price) if price is not None else 0.0,
                        float(ai_prob) if ai_prob is not None else 0.0,
                        float(sentiment) if sentiment is not None else 0.0,
                        str(reason) if reason is not None else "",
                        str(market_regime) if market_regime is not None else "",
                        1 if bool(is_crypto) else 0,
                        pj,
                    ),
                )
                conn.commit()
            except Exception as e:
                try:
                    from .log_throttle import log_exception_throttled
                    log_exception_throttled(
                        self._log,
                        "E_DB_LOG_DECISION_PACKET",
                        e,
                        key="db_log_decision_packet",
                        throttle_sec=300,
                        context={"symbol": symbol, "strategy": strategy, "action": action},
                    )
                except Exception:
                    pass

    def log_execution_packet(
        self,
        *,
        symbol: str,
        side: str,
        phase: str,
        decision_id: str = None,
        qty: float = None,
        price: float = None,
        order_id: str = None,
        client_order_id: str = None,
        broker_status: str = None,
        payload: dict = None,
        timestamp=None,
        event_id: str = None,
    ) -> None:
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        ts = timestamp or datetime.now()
        try:
            pj = json.dumps(payload or {}, ensure_ascii=False)
        except Exception:
            pj = "{}"
        eid = str(event_id) if event_id else str(__import__('uuid').uuid4())
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO execution_packets "
                    "(event_id, timestamp, decision_id, symbol, side, phase, qty, price, order_id, client_order_id, broker_status, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        eid,
                        ts,
                        str(decision_id) if decision_id else None,
                        str(symbol).upper(),
                        str(side),
                        str(phase),
                        float(qty) if qty is not None else None,
                        float(price) if price is not None else None,
                        str(order_id) if order_id else None,
                        str(client_order_id) if client_order_id else None,
                        str(broker_status) if broker_status else None,
                        pj,
                    ),
                )
                conn.commit()
            except Exception as e:
                try:
                    from .log_throttle import log_exception_throttled
                    log_exception_throttled(
                        self._log,
                        "E_DB_LOG_EXEC_PACKET",
                        e,
                        key="db_log_exec_packet",
                        throttle_sec=300,
                        context={"symbol": symbol, "side": side, "phase": phase, "order_id": order_id, "client_order_id": client_order_id},
                    )
                except Exception:
                    pass

    def get_decision_packets(self, limit: int = 500, symbol: str = None):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        try:
            lim = int(limit) if limit is not None else 500
        except Exception:
            lim = 500
        if lim < 1:
            lim = 1
        if lim > 5000:
            lim = 5000
        with lock:
            try:
                cur = conn.cursor()
                if symbol:
                    cur.execute(
                        "SELECT decision_id, timestamp, symbol, strategy, action, score, price, ai_prob, sentiment, reason, market_regime, is_crypto, payload_json "
                        "FROM decision_packets WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                        (str(symbol).upper(), lim),
                    )
                else:
                    cur.execute(
                        "SELECT decision_id, timestamp, symbol, strategy, action, score, price, ai_prob, sentiment, reason, market_regime, is_crypto, payload_json "
                        "FROM decision_packets ORDER BY timestamp DESC LIMIT ?",
                        (lim,),
                    )
                return cur.fetchall() or []
            except Exception:
                return []

    def get_execution_packets(self, limit: int = 2000, symbol: str = None, decision_id: str = None):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        try:
            lim = int(limit) if limit is not None else 2000
        except Exception:
            lim = 2000
        if lim < 1:
            lim = 1
        if lim > 20000:
            lim = 20000
        with lock:
            try:
                cur = conn.cursor()
                if decision_id:
                    cur.execute(
                        "SELECT event_id, timestamp, decision_id, symbol, side, phase, qty, price, order_id, client_order_id, broker_status, payload_json "
                        "FROM execution_packets WHERE decision_id=? ORDER BY timestamp ASC LIMIT ?",
                        (str(decision_id), lim),
                    )
                elif symbol:
                    cur.execute(
                        "SELECT event_id, timestamp, decision_id, symbol, side, phase, qty, price, order_id, client_order_id, broker_status, payload_json "
                        "FROM execution_packets WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                        (str(symbol).upper(), lim),
                    )
                else:
                    cur.execute(
                        "SELECT event_id, timestamp, decision_id, symbol, side, phase, qty, price, order_id, client_order_id, broker_status, payload_json "
                        "FROM execution_packets ORDER BY timestamp DESC LIMIT ?",
                        (lim,),
                    )
                return cur.fetchall() or []
            except Exception:
                return []

    # --------------------
    # v5.12.7 updateA: Config history (Config tab foundation)
    # --------------------
    def log_config_change(self, *, cfg_key: str, old_value: str, new_value: str, source: str = "manual", timestamp=None) -> None:
        """Persist a config change into config_history (best-effort).

        Notes:
          - Stored in decision_logs.db to avoid new DB files.
          - Controlled by CONFIGURATION->config_history_enabled.
        """
        try:
            if not self._cfg_bool("config_history_enabled", True):
                return
        except Exception:
            pass

        if not cfg_key:
            return

        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        ts = timestamp or datetime.now()

        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO config_history (timestamp, cfg_key, old_value, new_value, source) VALUES (?, ?, ?, ?, ?)",
                    (ts, str(cfg_key), str(old_value), str(new_value), str(source)),
                )
                conn.commit()
            except Exception:
                return

            # Prune old rows if needed
            try:
                max_rows = 5000
                try:
                    raw = None
                    if self.config:
                        raw = self.config.get("CONFIGURATION", "config_history_max_rows", fallback=str(max_rows))
                    if raw is not None:
                        max_rows = int(str(raw).strip())
                except Exception:
                    max_rows = 5000
                if max_rows < 100:
                    max_rows = 100
                if max_rows > 500000:
                    max_rows = 500000

                cur = conn.cursor()
                row = cur.execute("SELECT COUNT(1) FROM config_history").fetchone()
                n = int(row[0]) if row and row[0] is not None else 0
                if n > max_rows:
                    # Delete oldest extra rows
                    to_del = n - max_rows
                    cur.execute(
                        "DELETE FROM config_history WHERE id IN (SELECT id FROM config_history ORDER BY timestamp ASC LIMIT ?)",
                        (to_del,),
                    )
                    conn.commit()
            except Exception:
                pass

    # --------------------
    # v5.13.1 updateB: Watchlist policy audit log
    # --------------------
    def log_watchlist_audit(
        self,
        *,
        batch_id: str,
        source: str,
        mode: str,
        before: List[str],
        after: List[str],
        added: List[str],
        removed: List[str],
        meta: Optional[Dict[str, Any]] = None,
        timestamp=None,
    ) -> None:
        """Write a watchlist policy run to watchlist_audit (best-effort)."""

        if not batch_id:
            return

        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        ts = timestamp or datetime.utcnow()

        try:
            before_json = json.dumps(list(before or []))
            after_json = json.dumps(list(after or []))
            added_json = json.dumps(list(added or []))
            removed_json = json.dumps(list(removed or []))
            meta_json = json.dumps(meta or {})
        except Exception:
            return

        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT OR REPLACE INTO watchlist_audit (batch_id, timestamp, source, mode, before_json, after_json, added_json, removed_json, meta_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(batch_id),
                        ts,
                        str(source or ""),
                        str(mode or ""),
                        before_json,
                        after_json,
                        added_json,
                        removed_json,
                        meta_json,
                    ),
                )
                conn.commit()
            except Exception:
                return

    def get_config_history(self, limit: int = 200):
        """Return recent config changes.

        Returns tuples:
          (timestamp, cfg_key, old_value, new_value, source)
        """
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        try:
            lim = int(limit) if limit is not None else 200
        except Exception:
            lim = 200
        if lim < 1:
            lim = 1
        if lim > 5000:
            lim = 5000
        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT timestamp, cfg_key, old_value, new_value, source FROM config_history ORDER BY timestamp DESC LIMIT ?",
                    (lim,),
                )
                return cur.fetchall() or []
            except Exception:
                return []

    def get_recent_decisions(self, limit: int = 100):
        """Return recent decision logs for UI viewers.

        Returns tuples:
          (timestamp, symbol, action, rsi, ai_score, reason)
        """
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        try:
            lim = int(limit) if limit is not None else 100
        except Exception:
            lim = 100
        if lim < 1:
            lim = 1
        if lim > 1000:
            lim = 1000

        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT timestamp, symbol, action, rsi, ai_score, reason "
                    "FROM decision_logs ORDER BY timestamp DESC LIMIT ?",
                    (lim,)
                )
                return cur.fetchall() or []
            except Exception as e:
                try:
                    from .log_throttle import log_exception_throttled
                    log_exception_throttled(
                        self._log,
                        "E_DB_GET_DECISIONS",
                        e,
                        key="db_get_decisions",
                        throttle_sec=300,
                        context={"limit": lim},
                    )
                except Exception:
                    pass
                return []

    # --- v5.13.1 updateA: CANDIDATE SCANNER ---
    def save_candidates(self, scan_id: str, rows: List[Dict[str, Any]]) -> int:
        """Insert candidate scan rows into the candidates table."""
        if not rows:
            return 0

        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")

        with lock:
            try:
                cur = conn.cursor()
                q = """INSERT OR REPLACE INTO candidates
                    (scan_id, scan_ts, scan_date, symbol, score, ret_lookback, dollar_volume, volatility, bars, universe, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                payload = []
                for r in rows:
                    payload.append((
                        scan_id,
                        r.get("scan_ts"),
                        r.get("scan_date"),
                        r.get("symbol"),
                        r.get("score"),
                        r.get("ret_lookback"),
                        r.get("dollar_volume"),
                        r.get("volatility"),
                        r.get("bars"),
                        r.get("universe"),
                        r.get("details_json"),
                    ))
                cur.executemany(q, payload)
                conn.commit()
                return len(payload)
            except Exception as e:
                try:
                    self._logger.exception("DB Error Saving Candidates")
                except Exception:
                    pass
                return 0


    def get_latest_candidates(self, *, scan_date: Optional[str] = None, limit: int = 50):
        """Return the latest candidates for a date (UTC) as a DataFrame."""
        try:
            if scan_date is None:
                scan_date = datetime.utcnow().strftime("%Y-%m-%d")
        except Exception:
            scan_date = ""

        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")

        with lock:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT scan_id FROM candidates WHERE scan_date=? ORDER BY scan_ts DESC LIMIT 1",
                    (scan_date,),
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    return pd.DataFrame()

                sid = row[0]
                q = """SELECT scan_id, scan_ts, scan_date, symbol, score, ret_lookback, dollar_volume, volatility, bars, universe, details_json
                       FROM candidates WHERE scan_id=? ORDER BY score DESC LIMIT ?"""
                return pd.read_sql_query(q, conn, params=(sid, int(limit)))
            except Exception:
                return pd.DataFrame()

    # --- v3.3: HEATMAP DATA ---
    def get_latest_snapshot(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return latest CLOSE price for the symbol (float) or None."""
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT close FROM historical_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
                (symbol.upper(),)
            )
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] is not None else None
    def rebuild_backtest_table(self, strategy_names):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS backtest_results")
                cols = ["symbol TEXT PRIMARY KEY"]
                for s in strategy_names:
                    cols.append(f"PL_{s} REAL")
                    cols.append(f"Trades_{s} INTEGER")
                cols.append("best_strategy TEXT")
                cols.append("best_profit REAL")
                cols.append("timestamp DATETIME")
                query = f"CREATE TABLE backtest_results ({', '.join(cols)})"
                cursor.execute(query)
                # Compatibility: legacy name used in earlier builds
                try:
                    cursor.execute("DROP VIEW IF EXISTS backtest_testing_data")
                    cursor.execute("CREATE VIEW backtest_testing_data AS SELECT * FROM backtest_results")
                except Exception:
                    pass
                # v5.12.0: useful for reports / UI queries
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_timestamp ON backtest_results(timestamp)")
                except Exception:
                    pass
                conn.commit()
            except Exception:
                self._logger.exception("DB Error Rebuilding Backtest Table")
    def ensure_backtest_table(self, strategy_names):
        """Ensure backtest_results exists and contains columns for given strategies.

        Unlike rebuild_backtest_table(), this does NOT drop existing results.
        """
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")

        with lock:
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'")
                exists = cur.fetchone() is not None

                base_cols = [
                    ("symbol", "TEXT PRIMARY KEY"),
                    ("best_strategy", "TEXT"),
                    ("best_profit", "REAL"),
                    ("timestamp", "DATETIME"),
                ]

                if not exists:
                    cols = [f"{n} {t}" for (n, t) in base_cols]
                    for s in strategy_names:
                        cols.append(f"PL_{s} REAL")
                        cols.append(f"Trades_{s} INTEGER")
                    cur.execute(f"CREATE TABLE backtest_results ({', '.join(cols)})")
                    try:
                        cur.execute("DROP VIEW IF EXISTS backtest_testing_data")
                        cur.execute("CREATE VIEW backtest_testing_data AS SELECT * FROM backtest_results")
                    except Exception:
                        pass
                    try:
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_timestamp ON backtest_results(timestamp)")
                    except Exception:
                        pass
                    conn.commit()
                    return

                cur.execute("PRAGMA table_info(backtest_results)")
                existing = {row[1] for row in cur.fetchall()}

                for s in strategy_names:
                    pl = f"PL_{s}"
                    tr = f"Trades_{s}"
                    if pl not in existing:
                        try:
                            cur.execute(f"ALTER TABLE backtest_results ADD COLUMN {pl} REAL")
                        except Exception:
                            pass
                    if tr not in existing:
                        try:
                            cur.execute(f"ALTER TABLE backtest_results ADD COLUMN {tr} INTEGER")
                        except Exception:
                            pass

                # Defensive base columns (can't add PK post-hoc)
                if "symbol" not in existing:
                    try:
                        cur.execute("ALTER TABLE backtest_results ADD COLUMN symbol TEXT")
                    except Exception:
                        pass
                if "best_strategy" not in existing:
                    try:
                        cur.execute("ALTER TABLE backtest_results ADD COLUMN best_strategy TEXT")
                    except Exception:
                        pass
                if "best_profit" not in existing:
                    try:
                        cur.execute("ALTER TABLE backtest_results ADD COLUMN best_profit REAL")
                    except Exception:
                        pass
                if "timestamp" not in existing:
                    try:
                        cur.execute("ALTER TABLE backtest_results ADD COLUMN timestamp DATETIME")
                    except Exception:
                        pass

                try:
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_timestamp ON backtest_results(timestamp)")
                except Exception:
                    pass
                conn.commit()
            except Exception as e:
                try:
                    self._logger.exception("DB Error Ensuring Backtest Table")
                except Exception:
                    pass

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
                self._logger.exception("DB Error Saving Backtest")
    def get_backtest_data(self):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            try:
                return pd.read_sql_query("SELECT * FROM backtest_results ORDER BY timestamp DESC", conn)
            except Exception:
                # Fallback for legacy schemas or partial tables (e.g. missing timestamp column)
                try:
                    return pd.read_sql_query("SELECT * FROM backtest_results", conn)
                except Exception:
                    return pd.DataFrame()
    def get_best_strategy_for_symbol(self, symbol):
        conn = self._conn("backtest_results")
        lock = self._lock("backtest_results")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT best_strategy, best_profit, timestamp FROM backtest_results WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
                (symbol.upper(),)
            )
            return cursor.fetchone()
    def get_all_symbols(self):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM historical_prices")
            rows = cursor.fetchall()
            return [r[0] for r in rows]

    def save_bulk_data(self, symbol, data_list):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Bulk insert minute bars into historical_prices.

        Returns the number of rows actually inserted (INSERT OR IGNORE semantics).
        """
        with lock:
            try:
                before = conn.total_changes
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR IGNORE INTO historical_prices 
                    (symbol, timestamp, close, open, high, low, volume, rsi, bb_lower, bb_upper, ema_200, adx, atr) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data_list)
                conn.commit()
                after = conn.total_changes
                return max(0, after - before)
            except Exception as e:
                self._logger.exception("DB Bulk Write Error")
                return 0
    def save_snapshot(self, symbol, timestamp, price, open_p, high, low, volume, rsi, bbl, bbu, ema, adx, atr):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        with lock:
            try:
                symbol = symbol.upper()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO historical_prices 
                    (symbol, timestamp, close, open, high, low, volume, rsi, bb_lower, bb_upper, ema_200, adx, atr) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timestamp, price, open_p, high, low, volume, rsi, bbl, bbu, ema, adx, atr))
                conn.commit()
            except Exception:
                self._logger.exception("DB Write Error")
    def get_last_timestamp(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return the most recent timestamp stored for a symbol in historical_prices (UTC-aware datetime) or None."""
        sym_u = symbol.upper()
        sym_l = symbol.lower()
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM historical_prices WHERE symbol=?", (sym_u,))
            row = cursor.fetchone()
            if (not row) or row[0] is None:
                if sym_l != sym_u:
                    cursor.execute("SELECT MAX(timestamp) FROM historical_prices WHERE symbol=?", (sym_l,))
                    row = cursor.fetchone()
                if (not row) or row[0] is None:
                    return None
            ts = row[0]
        try:
            dt = pd.to_datetime(ts, errors='coerce', utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            try:
                if isinstance(ts, datetime):
                    return ts
            except Exception:
                pass
            return None


    def get_first_timestamp(self, symbol):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return the oldest timestamp stored for a symbol in historical_prices (UTC-aware datetime) or None."""
        sym_u = symbol.upper()
        sym_l = symbol.lower()
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(timestamp) FROM historical_prices WHERE symbol=?", (sym_u,))
            row = cursor.fetchone()
            if (not row) or row[0] is None:
                if sym_l != sym_u:
                    cursor.execute("SELECT MIN(timestamp) FROM historical_prices WHERE symbol=?", (sym_l,))
                    row = cursor.fetchone()
                if (not row) or row[0] is None:
                    return None
            ts = row[0]
        try:
            dt = pd.to_datetime(ts, errors='coerce', utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            try:
                if isinstance(ts, datetime):
                    return ts
            except Exception:
                pass
            return None
    def get_history(self, symbol, limit=120):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return recent history for symbol sorted by timestamp (ascending)."""
        sym_u = symbol.upper()
        sym_l = symbol.lower()

        with lock:
            try:
                q = "SELECT * FROM historical_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT ?"
                df = pd.read_sql_query(q, conn, params=(sym_u, int(limit)))
                if df.empty and sym_l != sym_u:
                    df = pd.read_sql_query(q, conn, params=(sym_l, int(limit)))
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return df

        # v4.0.4B: Ensure timestamp and numeric columns are correct types
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        except Exception:
            pass

        for col in ['close', 'open', 'high', 'low', 'volume', 'rsi', 'bb_lower', 'bb_upper', 'ema_200', 'adx', 'atr']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        # Filter invalid OHLC rows (prevents extreme spikes from bad rows)
        try:
            df = df.dropna(subset=['open','high','low','close'])
            df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
            df = df[df['high'] >= df['low']]
        except Exception:
            pass

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    def get_trade_markers(self, symbol):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            try:
                df = pd.read_sql_query(
                    "SELECT entry_time, exit_time, entry_price, exit_price FROM trade_history WHERE symbol=?",
                    conn,
                    params=(symbol.upper(),)
                )
            except Exception:
                return None

        if df is None or df.empty:
            return None

        try:
            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        except Exception:
            pass
        return df

    def log_trade_entry(self, symbol, qty, price, strategy):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute('INSERT OR REPLACE INTO active_trades VALUES (?, ?, ?, ?, ?, ?)',
                            (symbol.upper(), qty, price, price, strategy, datetime.now()))
                conn.commit()
            except Exception:
                self._logger.exception("DB Trade Entry Error")
    def update_highest_price(self, symbol, new_high):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE active_trades SET highest_price = ? WHERE symbol = ?", (new_high, symbol.upper()))
                conn.commit()
            except Exception as e:
                # v5.12.3 updateA: don't silently swallow DB write failures.
                try:
                    from .log_throttle import log_exception_throttled
                    log_exception_throttled(
                        self._log,
                        "E_DB_UPDATE_HIGHEST_PRICE",
                        e,
                        key=f"db_highest_{str(symbol).upper()}",
                        throttle_sec=300,
                        context={"symbol": symbol, "new_high": new_high},
                    )
                except Exception:
                    pass
    def get_active_trades_rows(self):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        """Low-level: return active_trades rows as tuples (symbol, qty, entry_price, highest_price, strategy, timestamp)."""
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, qty, entry_price, highest_price, strategy, timestamp FROM active_trades")
            return cursor.fetchall()
    def get_active_trades(self):
        """Return active trades as a dict keyed by symbol for UI/engine consumption."""
        rows = self.get_active_trades_rows()
        trades = {}
        try:
            for row in rows or []:
                try:
                    sym = str(row[0]).upper() if row[0] is not None else None
                    if not sym:
                        continue
                    trades[sym] = {
                        'qty': int(row[1]) if row[1] is not None else 0,
                        'entry_price': float(row[2]) if row[2] is not None else 0.0,
                        'highest_price': float(row[3]) if row[3] is not None else (float(row[2]) if row[2] is not None else 0.0),
                        'strategy': str(row[4]) if row[4] is not None else 'UNKNOWN',
                        'timestamp': row[5],
                    }
                except Exception:
                    continue
        except Exception:
            return {}
        return trades

    def remove_active_trade(self, symbol):
        conn = self._conn("active_trades")
        lock = self._lock("active_trades")
        """Remove an active trade row without recording trade_history.
        Used for correcting DB state during syncs when a position does not exist.
        """
        with lock:
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))
                conn.commit()
            except Exception as e:
                self._logger.exception("DB Remove Active Trade Error")
    def close_trade(self, symbol, exit_price):
        if self.split_mode:
            with self._multi_lock(["active_trades", "trade_history"]):
                try:
                    at_conn = self._conn("active_trades")
                    th_conn = self._conn("trade_history")

                    at_cur = at_conn.cursor()
                    th_cur = th_conn.cursor()

                    at_cur.execute("SELECT * FROM active_trades WHERE symbol=?", (symbol.upper(),))
                    row = at_cur.fetchone()
                    if row:
                        qty = row[1]
                        entry_price = row[2]
                        strategy = row[4]
                        entry_time = row[5]
                        pl = (exit_price - entry_price) * qty

                        th_cur.execute('''
                            INSERT INTO trade_history (symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol.upper(), qty, entry_price, exit_price, pl, strategy, entry_time, datetime.now()))

                        at_cur.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))

                        th_conn.commit()
                        at_conn.commit()
                        return pl
                    return 0
                except Exception:
                    return 0

        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM active_trades WHERE symbol=?", (symbol.upper(),))
                row = cursor.fetchone()
                if row:
                    qty = row[1]
                    entry_price = row[2]
                    strategy = row[4]
                    entry_time = row[5]
                    pl = (exit_price - entry_price) * qty
                    cursor.execute('''
                        INSERT INTO trade_history (symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol.upper(), qty, entry_price, exit_price, pl, strategy, entry_time, datetime.now()))
                    cursor.execute("DELETE FROM active_trades WHERE symbol=?", (symbol.upper(),))
                    self.conn.commit()
                    return pl
                return 0
            except Exception:
                return 0

    def get_portfolio_stats(self):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        """Return portfolio stats as a dict used by the Portfolio tab."""
        with lock:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(profit_loss) FROM trade_history")
            total_trades, total_profit = cursor.fetchone()
            total_trades = int(total_trades or 0)
            total_profit = float(total_profit or 0.0)
            cursor.execute("SELECT COUNT(*) FROM trade_history WHERE profit_loss > 0")
            wins = int((cursor.fetchone() or [0])[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM trade_history WHERE profit_loss < 0")
            losses = int((cursor.fetchone() or [0])[0] or 0)
            win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
            return {
                'total_trades': total_trades,
                'total_pl': total_profit,
                'wins': wins,
                'losses': losses,
                'win_rate': float(win_rate),
            }
    def get_strategy_stats(self):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT strategy, COUNT(*), SUM(profit_loss) FROM trade_history GROUP BY strategy"
            )
            return cursor.fetchall()
    def get_recent_history(self, limit=50):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time "
                "FROM trade_history ORDER BY exit_time DESC LIMIT ?",
                (int(limit),)
            )
            return cursor.fetchall()
    def get_trade_history_since(self, since_dt):
        conn = self._conn("trade_history")
        lock = self._lock("trade_history")
        """Return trade_history rows with exit_time >= since_dt.

        Args:
            since_dt: datetime or ISO string
        Returns:
            list[dict]
        """
        try:
            if isinstance(since_dt, datetime):
                since_str = since_dt.isoformat(sep=' ', timespec='seconds')
            else:
                since_str = str(since_dt)

            with lock:
                cur = conn.cursor()
                cur.execute(
                    "SELECT symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time "
                    "FROM trade_history WHERE exit_time >= ? ORDER BY exit_time ASC",
                    (since_str,)
                )
                rows = cur.fetchall()

            out = []
            for r in rows:
                out.append({
                    'symbol': r[0],
                    'qty': r[1],
                    'entry_price': r[2],
                    'exit_price': r[3],
                    'profit_loss': r[4],
                    'strategy': r[5],
                    'entry_time': r[6],
                    'exit_time': r[7],
                })
            return out
        except Exception:
            return []
    def get_decision_counts_since(self, since_dt):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        """Return counts of decision log actions since since_dt."""
        try:
            if isinstance(since_dt, datetime):
                since_str = since_dt.isoformat(sep=' ', timespec='seconds')
            else:
                since_str = str(since_dt)

            with lock:
                cur = conn.cursor()
                cur.execute(
                    "SELECT action, COUNT(1) FROM decision_logs WHERE timestamp >= ? GROUP BY action",
                    (since_str,)
                )
                rows = cur.fetchall()
            return {str(a): int(c) for (a, c) in rows}
        except Exception:
            return {}
    def get_latest_timestamps_for_symbols(self, symbols):
        conn = self._conn("historical_prices")
        lock = self._lock("historical_prices")
        """Return {SYMBOL: latest_timestamp_str} for symbols present in historical_prices."""
        try:
            syms = [s.upper() for s in (symbols or []) if s]
            if not syms:
                return {}

            placeholders = ','.join(['?'] * len(syms))
            sql = (
                f"SELECT symbol, MAX(timestamp) as ts FROM historical_prices "
                f"WHERE UPPER(symbol) IN ({placeholders}) GROUP BY symbol"
            )

            with lock:
                cur = conn.cursor()
                cur.execute(sql, syms)
                rows = cur.fetchall()

            return {str(sym).upper(): (str(ts) if ts is not None else None) for sym, ts in rows}
        except Exception:
            return {}


    # ---------------- Agent Shadow Mode (v5.13.0 Update A) -----------------

    def _agent_fingerprint(self, title: str, suggestion_type: str, suggestion_json: str, artifact_path: str) -> str:
        raw = f"{title}|{suggestion_type}|{artifact_path}|{suggestion_json}".encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()

    def upsert_agent_suggestion(self, artifact_type: str, artifact_path: str, title: str, suggestion_type: str, suggestion_payload: dict, status: str = "NEW"):
        """Insert a new suggestion if fingerprint doesn't exist; returns suggestion_id."""
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

    def add_agent_rationale(self, suggestion_id: int, rationale: str, metrics_payload: dict | None = None):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_json = json.dumps(metrics_payload or {}, ensure_ascii=False)
        with lock:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO agent_rationales (suggestion_id, created_at, rationale, metrics_json) VALUES (?, ?, ?, ?)",
                (int(suggestion_id), created_at, rationale, metrics_json),
            )
            conn.commit()

    def get_agent_suggestions(self, limit: int = 200, status: str | None = None):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        q = "SELECT id, created_at, title, suggestion_type, status, artifact_type, artifact_path FROM agent_suggestions"
        params = []
        if status:
            q += " WHERE status = ?"
            params.append(status)
        q += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with lock:
            cur = conn.cursor()
            cur.execute(q, tuple(params))
            rows = cur.fetchall()
        return rows

    def get_agent_suggestion_detail(self, suggestion_id: int):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        with lock:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, created_at, title, suggestion_type, status, artifact_type, artifact_path, suggestion_json, applied_at, applied_by FROM agent_suggestions WHERE id = ?",
                (int(suggestion_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                "SELECT created_at, rationale, metrics_json FROM agent_rationales WHERE suggestion_id = ? ORDER BY id ASC",
                (int(suggestion_id),),
            )
            rats = cur.fetchall()
        return row, rats

    def set_agent_suggestion_status(self, suggestion_id: int, status: str, applied_by: str | None = None):
        conn = self._conn("decision_logs")
        lock = self._lock("decision_logs")
        applied_at = None
        if status.upper() in {"APPLIED", "IGNORED"}:
            applied_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with lock:
            cur = conn.cursor()
            cur.execute(
                "UPDATE agent_suggestions SET status = ?, applied_at = COALESCE(?, applied_at), applied_by = COALESCE(?, applied_by) WHERE id = ?",
                (status, applied_at, applied_by, int(suggestion_id)),
            )
            conn.commit()
