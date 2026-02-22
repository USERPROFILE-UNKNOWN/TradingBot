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


DEFAULT_PLATFORM = "alpaca"


def _discover_platforms(config_dir: str) -> list[str]:
    """Return discovered platform ids from config/<platform>/ folders."""
    found: list[str] = []
    try:
        if os.path.isdir(config_dir):
            for name in sorted(os.listdir(config_dir)):
                p = os.path.join(config_dir, name)
                if not os.path.isdir(p):
                    continue
                if name.startswith("_"):
                    continue
                # Platform folders carry platform-specific INI files.
                if os.path.isfile(os.path.join(p, "keys.ini")) or os.path.isfile(os.path.join(p, "watchlist.ini")):
                    found.append(name)
    except Exception:
        return [DEFAULT_PLATFORM]

    if not found:
        return [DEFAULT_PLATFORM]
    return found


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
        if os.path.isdir(mod_dir) and os.path.isfile(os.path.join(mod_dir, "main.py")):
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
    db_root_dir = os.path.join(root_path, "db")
    logs_root_dir = os.path.join(root_path, "logs")
    backup_root_dir = os.path.join(root_path, "backups")

    available_platforms = _discover_platforms(config_dir)
    selected_platform = (os.environ.get("TRADINGBOT_PLATFORM") or "").strip().lower() or DEFAULT_PLATFORM
    if selected_platform not in available_platforms:
        selected_platform = DEFAULT_PLATFORM if DEFAULT_PLATFORM in available_platforms else available_platforms[0]

    platform_config_dir = os.path.join(config_dir, selected_platform)
    platform_db_dir = os.path.join(db_root_dir, selected_platform)
    platform_logs_dir = os.path.join(logs_root_dir, selected_platform)
    platform_backup_dir = os.path.join(backup_root_dir, selected_platform)

    paths = {
        # canonical root keys
        "root": root_path,

        # config
        "config_dir": config_dir,
        "platform": selected_platform,
        "available_platforms": available_platforms,
        "platform_config_dir": platform_config_dir,
        "keys_ini": os.path.join(platform_config_dir, "keys.ini"),
        "configuration_ini": os.path.join(config_dir, "config.ini"),
        "sectors_ini": os.path.join(config_dir, "sectors.ini"),
        "watchlist_ini": os.path.join(platform_config_dir, "watchlist.ini"),
        "strategy_ini": os.path.join(config_dir, "strategy.ini"),

        # db / logs / backups
        "db_root_dir": db_root_dir,
        "db_dir": platform_db_dir,
        "logs_root_dir": logs_root_dir,
        "logs": platform_logs_dir,
        "backup_root_dir": backup_root_dir,
        "backup": platform_backup_dir,
    }

    return paths
