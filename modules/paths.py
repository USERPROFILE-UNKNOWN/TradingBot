"""Path discovery utilities.

This module centralizes the "where am I running from" logic.

Contract:
- Works for both source runs and PyInstaller-frozen executables.
- Returns a dict with stable keys used across the project.

"""

from __future__ import annotations

import os
import sys
from typing import Dict


def get_paths() -> Dict[str, str]:
    """Determine runtime paths (source vs frozen) and return a path map."""

    def looks_like_root(p: str) -> bool:
        if not p:
            return False
        try:
            if not os.path.isdir(p):
                return False
        except Exception:
            return False

        cfg_dir = os.path.join(p, "config")
        mod_dir = os.path.join(p, "modules")

        # Canonical layout: <root>/config + <root>/modules
        if os.path.isdir(cfg_dir) and os.path.isdir(mod_dir):
            if os.path.isfile(os.path.join(cfg_dir, "config.ini")):
                return True
            if os.path.isfile(os.path.join(cfg_dir, "watchlist.ini")):
                return True
            if os.path.isfile(os.path.join(cfg_dir, "keys.ini")):
                return True

        # Another common root marker
        if os.path.isdir(mod_dir) and os.path.isfile(os.path.join(p, "main.py")):
            return True

        return False

    # --- Base starting point ---
    if getattr(sys, "frozen", False):
        # Running as a compiled .exe
        start_dir = os.path.dirname(sys.executable)
    else:
        # Running from source; this file is in <root>/modules
        module_path = os.path.dirname(os.path.abspath(__file__))
        start_dir = os.path.dirname(module_path)

    # --- Optional override ---
    env_root = (os.environ.get("TRADINGBOT_ROOT") or os.environ.get("TRADINGBOT_HOME") or "").strip()
    if env_root:
        env_root = os.path.abspath(env_root)
        if looks_like_root(env_root):
            root_path = env_root
        else:
            root_path = start_dir
    else:
        root_path = start_dir

    # --- Probe a few nearby candidates (helps PyInstaller dist layouts) ---
    try:
        candidates = []
        base = os.path.abspath(root_path)
        parents = [base]
        try:
            parents.append(os.path.dirname(parents[-1]))
            parents.append(os.path.dirname(parents[-1]))
        except Exception:
            pass

        for p in parents:
            if not p:
                continue
            candidates.append(p)
            candidates.append(os.path.join(p, "TradingBot"))

        seen = set()
        for c in candidates:
            if not c:
                continue
            c = os.path.abspath(c)
            if c in seen:
                continue
            seen.add(c)
            if looks_like_root(c):
                root_path = c
                break
    except Exception:
        pass

    root_path = os.path.abspath(root_path)

    config_dir = os.path.join(root_path, "config")
    db_dir = os.path.join(root_path, "db")
    logs_dir = os.path.join(root_path, "logs")
    backup_dir = os.path.join(root_path, "backups")

    paths = {
        # canonical root keys
        "root": root_path,

        # config
        "config_dir": config_dir,
        "keys_ini": os.path.join(config_dir, "keys.ini"),
        "configuration_ini": os.path.join(config_dir, "config.ini"),
        "watchlist_ini": os.path.join(config_dir, "watchlist.ini"),
        "strategy_ini": os.path.join(config_dir, "strategy.ini"),

        # db / logs / backups
        "db_dir": db_dir,
        "logs": logs_dir,
        "backup": backup_dir,
    }

    return paths
