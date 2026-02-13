import configparser
import os
import sys
import types

# Avoid optional heavy dependency during test collection.
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from modules.startup_paths import resolve_db_placeholder_path
from modules.database import DataManager


def _cfg(db_dir=None):
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {}
    if db_dir is not None:
        c["CONFIGURATION"]["db_dir"] = db_dir
    return c


def test_resolve_db_placeholder_path_uses_config_dir_for_relative():
    paths = {"root": "/app", "config_dir": "/app/config", "db_dir": "/app/db"}
    cfg = _cfg("../runtime_db")
    got = resolve_db_placeholder_path(paths, cfg)
    assert got == os.path.normpath("/app/runtime_db/market_data.db")


def test_resolve_db_placeholder_path_falls_back_to_paths_db_dir():
    paths = {"root": "/app", "db_dir": "/app/db"}
    cfg = _cfg("")
    got = resolve_db_placeholder_path(paths, cfg)
    assert got == os.path.normpath("/app/db/market_data.db")


def test_datamanager_read_db_dir_relative_to_config_dir():
    dm = DataManager.__new__(DataManager)
    cfg = _cfg("../db_live")
    paths = {"config_dir": "/app/config"}
    got = dm._read_db_dir("/ignored/market_data.db", cfg, paths)
    assert got == os.path.normpath("/app/db_live")
