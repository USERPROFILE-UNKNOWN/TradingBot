import configparser
import os
import sys
import types

sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from modules.database import DataManager


def test_datamanager_initializes_split_db_files_under_db_dir(tmp_path):
    root = tmp_path / "TradingBot"
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True)

    # Use a relative db_dir so we validate path resolution against config_dir.
    cfg = configparser.ConfigParser()
    cfg["CONFIGURATION"] = {
        "agent_mode": "OFF",
        "db_dir": "../db_runtime",
    }
    cfg["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}

    paths = {
        "config_dir": str(cfg_dir),
        "backup": str(root / "backups"),
        "db_dir": str(root / "db"),
    }

    dm = None
    try:
        dm = DataManager(db_path=str(root / "db" / "market_data.db"), config=cfg, paths=paths)
        assert dm.split_mode is True
        assert os.path.normpath(dm.db_dir) == os.path.normpath(str(root / "db_runtime"))

        expected_files = {
            "historical_prices.db",
            "active_trades.db",
            "trade_history.db",
            "decision_logs.db",
            "backtest_results.db",
        }

        found = {os.path.basename(p) for p in dm.db_paths.values()}
        assert expected_files.issubset(found)
        for p in dm.db_paths.values():
            assert os.path.exists(p)

    finally:
        # Avoid leaving open handles on Windows during local runs.
        if dm is not None:
            try:
                for c in (dm.conns or {}).values():
                    try:
                        c.close()
                    except Exception:
                        pass
            except Exception:
                pass
