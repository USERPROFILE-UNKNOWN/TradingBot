import configparser
import os
import sys
import types

# Avoid optional heavy dependency during test collection.
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from modules.startup_paths import resolve_db_placeholder_path
from modules.database import DataManager
from modules.config_io import _repair_legacy_db_folder_layout


def _cfg(db_dir=None):
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {}
    if db_dir is not None:
        c["CONFIGURATION"]["db_dir"] = db_dir
    return c


def test_resolve_db_placeholder_path_uses_root_for_relative():
    paths = {"root": "/app", "config_dir": "/app/config", "db_dir": "/app/db"}
    cfg = _cfg("../runtime_db")
    got = resolve_db_placeholder_path(paths, cfg)
    assert got == os.path.normpath("/runtime_db/market_data.db")


def test_resolve_db_placeholder_path_falls_back_to_paths_db_dir():
    paths = {"root": "/app", "db_dir": "/app/db"}
    cfg = _cfg("")
    got = resolve_db_placeholder_path(paths, cfg)
    assert got == os.path.normpath("/app/db/market_data.db")


def test_datamanager_read_db_dir_relative_to_root():
    dm = DataManager.__new__(DataManager)
    cfg = _cfg("../db_live")
    paths = {"root": "/app"}
    got = dm._read_db_dir("/ignored/market_data.db", cfg, paths)
    assert got == os.path.normpath("/db_live")


def test_repair_legacy_db_folder_layout_moves_config_db_files(tmp_path):
    root = tmp_path / "TradingBot"
    config_db = root / "config" / "db"
    canonical_db = root / "db"
    config_db.mkdir(parents=True)
    (config_db / "trade_history.db").write_text("x", encoding="utf-8")

    paths = {"root": str(root), "config_dir": str(root / "config"), "db_dir": str(canonical_db)}
    _repair_legacy_db_folder_layout(paths)

    assert (canonical_db / "trade_history.db").exists()
    assert not (config_db / "trade_history.db").exists()
