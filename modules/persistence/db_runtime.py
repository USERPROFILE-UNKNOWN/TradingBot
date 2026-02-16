import os
import sqlite3
import threading
import shutil
from datetime import datetime
from contextlib import ExitStack, contextmanager
from typing import Callable, Dict, Optional


from ..logging_utils import get_logger


class DbRuntime:
    """Centralized SQLite runtime for the split-DB layout.

    Responsibilities (Phase 2.5 / v5.16.0):
      - Open + configure per-DB connections
      - Maintain per-DB locks
      - Apply conservative PRAGMA settings
      - Handle known schema corruption edge-case for backtest_results.db (forward-only repair)
    """

    def __init__(
        self,
        db_dir: str,
        *,
        paths: Optional[dict] = None,
        logger=None,
        read_agent_mode: Optional[Callable[[], str]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.db_dir = os.path.normpath(db_dir)
        self.paths = paths or {}
        self._logger = logger or get_logger(__name__)
        self._read_agent_mode = read_agent_mode or (lambda: "OFF")
        self._log_fn = log_fn

        self.db_paths: Dict[str, str] = {}
        self.conns: Dict[str, sqlite3.Connection] = {}
        self.locks: Dict[str, threading.Lock] = {}

        self.primary_key = "historical_prices"

    def _log(self, msg: str) -> None:
        # Best-effort, structured when possible.
        try:
            if callable(self._log_fn):
                self._log_fn(str(msg))
                return
        except Exception:
            pass

        try:
            self._logger.info(str(msg), extra={"component": "db", "mode": self._read_agent_mode()})
        except Exception:
            pass

    def configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply conservative PRAGMA settings for better concurrency + stability.

        Notes:
          - WAL improves read/write concurrency (especially updater writes + UI reads).
          - synchronous=NORMAL is a recommended pairing with WAL for performance while remaining safe.
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

    def _open_connection(self, key: str, path: str) -> sqlite3.Connection:
        """Open a sqlite connection with hardening + known repair behaviors."""
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            self.configure_connection(conn)
            # Force schema load early so we can detect malformed schema (e.g. bad views)
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            return conn
        except sqlite3.DatabaseError as e:
            msg = str(e)

            # Known issue (v5.10.0 migration): a view was created with a schema qualifier
            # (e.g. bt.backtest_results). That can render the DB schema 'malformed'.
            if key == "backtest_results" and (
                "malformed database schema" in msg
                or "cannot reference objects in database" in msg
                or "backtest_testing_data" in msg
            ):
                try:
                    ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
                    backup_root = self.paths.get("backup") or os.path.join(os.path.dirname(self.db_dir), "backups")
                    bdir = os.path.join(backup_root, "db_repairs", ts)
                    os.makedirs(bdir, exist_ok=True)
                    shutil.copy2(path, os.path.join(bdir, os.path.basename(path)))
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    try:
                        self._logger.warning(
                            "[E_DB_BACKTEST_SCHEMA_REPAIRED] backtest_results.db schema was malformed; backed up to: %s",
                            bdir,
                            extra={"component": "db", "mode": self._read_agent_mode()},
                        )
                    except Exception:
                        pass
                except Exception:
                    try:
                        self._logger.exception(
                            "[E_DB_BACKTEST_SCHEMA_REPAIR_FAIL] Failed repairing malformed backtest_results.db",
                            extra={"component": "db", "mode": self._read_agent_mode()},
                        )
                    except Exception:
                        pass

                conn = sqlite3.connect(path, check_same_thread=False)
                self.configure_connection(conn)
                return conn

            # For other DBs, re-raise (should not happen in normal operation)
            raise

    def init_split_connections(self) -> None:
        """Open one SQLite connection per logical table group."""
        self.db_paths = {
            "historical_prices": os.path.join(self.db_dir, "historical_prices.db"),
            "active_trades": os.path.join(self.db_dir, "active_trades.db"),
            "trade_history": os.path.join(self.db_dir, "trade_history.db"),
            "decision_logs": os.path.join(self.db_dir, "decision_logs.db"),
            "backtest_results": os.path.join(self.db_dir, "backtest_results.db"),
        }

        for key, path in self.db_paths.items():
            conn = self._open_connection(key, path)
            self.conns[key] = conn
            self.locks[key] = threading.Lock()

    @property
    def primary_conn(self) -> Optional[sqlite3.Connection]:
        return self.conns.get(self.primary_key)

    @property
    def primary_lock(self) -> threading.Lock:
        return self.locks.get(self.primary_key) or threading.Lock()

    def conn(self, key: str) -> sqlite3.Connection:
        return self.conns[key]

    def lock(self, key: str) -> threading.Lock:
        return self.locks.get(key, self.primary_lock)

    @contextmanager
    def multi_lock(self, keys):
        """Acquire multiple locks in a deterministic order to avoid deadlocks."""
        uniq = sorted(set(keys))
        with ExitStack() as stack:
            for k in uniq:
                lk = self.locks.get(k)
                if lk is None:
                    lk = threading.Lock()
                    self.locks[k] = lk
                stack.enter_context(lk)
            yield
