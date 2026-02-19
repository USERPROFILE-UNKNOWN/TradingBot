"""Startup path helpers isolated for testability."""

from __future__ import annotations

import os


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
