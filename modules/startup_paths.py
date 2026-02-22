"""Startup path helpers isolated for testability."""

from __future__ import annotations

import os
import shutil
from typing import Callable, Dict


_SPLIT_DB_FILENAMES = (
    "historical_prices.db",
    "active_trades.db",
    "trade_history.db",
    "decision_logs.db",
    "backtest_results.db",
)


def resolve_db_placeholder_path(paths, config) -> str:
    """Resolve placeholder db_path used only for call-site compatibility."""
    try:
        raw = (config.get("CONFIGURATION", "db_dir", fallback="") or "").strip()
    except Exception:
        raw = ""

    db_dir = ""
    if raw:
        try:
            raw = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            pass
        if os.path.isabs(raw):
            db_dir = raw
        else:
            base = paths.get("root") or os.getcwd()
            db_dir = os.path.normpath(os.path.join(base, raw))

    if not db_dir:
        db_dir = paths.get("db_dir") or os.path.join(paths.get("root", os.getcwd()), "db")

    return os.path.join(os.path.normpath(db_dir), "market_data.db")


def migrate_root_db_to_platform(paths: Dict[str, str], log_fn: Callable[[str], None] | None = None) -> int:
    """One-time migration: move root db/*.db into platform db dir when platform dir is empty.

    Returns the number of DB files moved.
    """

    def _log(msg: str) -> None:
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass

    try:
        root = (paths or {}).get("root") or os.getcwd()
        legacy_db_dir = os.path.join(root, "db")
        platform_db_dir = (paths or {}).get("db_dir") or legacy_db_dir

        legacy_db_dir = os.path.normpath(legacy_db_dir)
        platform_db_dir = os.path.normpath(platform_db_dir)

        if legacy_db_dir == platform_db_dir:
            return 0
        if not os.path.isdir(legacy_db_dir):
            return 0

        os.makedirs(platform_db_dir, exist_ok=True)

        # Only migrate when platform db dir is missing split DBs to avoid partial forks.
        target_has_split = any(os.path.exists(os.path.join(platform_db_dir, fn)) for fn in _SPLIT_DB_FILENAMES)
        if target_has_split:
            return 0

        moved = 0
        for name in sorted(os.listdir(legacy_db_dir)):
            src = os.path.join(legacy_db_dir, name)
            dst = os.path.join(platform_db_dir, name)
            if not os.path.isfile(src):
                continue
            if not name.lower().endswith(".db"):
                continue
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
            moved += 1

        if moved:
            _log(f"[DB_MIGRATION] Moved {moved} DB file(s) from {legacy_db_dir} -> {platform_db_dir}")
        return moved
    except Exception:
        return 0
