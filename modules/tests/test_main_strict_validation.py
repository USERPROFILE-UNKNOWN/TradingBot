import configparser
import logging
import importlib
import sys
import types

import pytest


def _cfg_strict_invalid():
    c = configparser.ConfigParser()
    c["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}
    c["CONFIGURATION"] = {
        "strict_config_validation": "True",
        "amount_to_trade": "0",  # invalid
        "max_positions": "5",
        "max_percent_per_stock": "0.2",
        "max_daily_loss": "100",
        "update_interval_sec": "60",
        "live_entry_ttl_seconds": "120",
        "db_dir": "../db",
    }
    return c


def test_main_raises_on_strict_validation_errors(monkeypatch, tmp_path, caplog):
    # Keep import lightweight: fake heavy/GUI modules before importing main.
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

    fake_ui = types.ModuleType("modules.ui")

    class DummyTradingApp:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("TradingApp should not be created when strict validation fails")

    fake_ui.TradingApp = DummyTradingApp
    monkeypatch.setitem(sys.modules, "modules.ui", fake_ui)

    main = importlib.import_module("main")
    monkeypatch.setattr(main, "TradingApp", DummyTradingApp)

    logs = tmp_path / "logs"
    backup = tmp_path / "backup"
    root = tmp_path / "root"

    monkeypatch.setattr(main, "get_paths", lambda: {
        "logs": str(logs),
        "backup": str(backup),
        "root": str(root),
        "config_dir": str(root / "config"),
        "db_dir": str(root / "db"),
    })
    monkeypatch.setattr(main, "ensure_split_config_layout", lambda _paths: None)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _cfg_strict_invalid())
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Configuration validation failed in strict mode"):
            main.main()

    assert any("amount_to_trade must be > 0" in rec.message for rec in caplog.records)



def _cfg_non_strict_invalid():
    c = configparser.ConfigParser()
    c["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}
    c["CONFIGURATION"] = {
        "strict_config_validation": "False",
        "amount_to_trade": "0",  # invalid but allowed in non-strict startup
        "max_positions": "5",
        "max_percent_per_stock": "0.2",
        "max_daily_loss": "100",
        "update_interval_sec": "60",
        "live_entry_ttl_seconds": "120",
        "db_dir": "../db",
    }
    return c


def test_main_continues_on_non_strict_validation_errors(monkeypatch, tmp_path, caplog):
    # Keep import lightweight: fake heavy/GUI modules before importing main.
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

    fake_ui = types.ModuleType("modules.ui")

    app_created = {"value": False}

    class DummyTradingApp:
        def __init__(self, *_args, **_kwargs):
            app_created["value"] = True

        def mainloop(self):
            return None

    fake_ui.TradingApp = DummyTradingApp
    monkeypatch.setitem(sys.modules, "modules.ui", fake_ui)

    main = importlib.import_module("main")
    monkeypatch.setattr(main, "TradingApp", DummyTradingApp)

    logs = tmp_path / "logs"
    backup = tmp_path / "backup"
    root = tmp_path / "root"

    monkeypatch.setattr(main, "get_paths", lambda: {
        "logs": str(logs),
        "backup": str(backup),
        "root": str(root),
        "config_dir": str(root / "config"),
        "db_dir": str(root / "db"),
    })
    monkeypatch.setattr(main, "ensure_split_config_layout", lambda _paths: None)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _cfg_non_strict_invalid())
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    # Avoid actual DB I/O in this startup behavior test.
    class DummyDataManager:
        def __init__(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(main, "DataManager", DummyDataManager)

    with caplog.at_level(logging.ERROR):
        main.main()

    assert app_created["value"] is True
    assert any("amount_to_trade must be > 0" in rec.message for rec in caplog.records)


def test_main_logs_when_sanitizer_report_access_fails(monkeypatch, tmp_path, caplog):
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

    fake_ui = types.ModuleType("modules.ui")

    class DummyTradingApp:
        def __init__(self, *_args, **_kwargs):
            return None

        def mainloop(self):
            return None

    fake_ui.TradingApp = DummyTradingApp
    monkeypatch.setitem(sys.modules, "modules.ui", fake_ui)

    main = importlib.import_module("main")
    monkeypatch.setattr(main, "TradingApp", DummyTradingApp)

    logs = tmp_path / "logs"
    backup = tmp_path / "backup"
    root = tmp_path / "root"

    monkeypatch.setattr(main, "get_paths", lambda: {
        "logs": str(logs),
        "backup": str(backup),
        "root": str(root),
        "config_dir": str(root / "config"),
        "db_dir": str(root / "db"),
    })
    monkeypatch.setattr(main, "ensure_split_config_layout", lambda _paths: None)

    cfg = configparser.ConfigParser()
    cfg["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}
    cfg["CONFIGURATION"] = {
        "strict_config_validation": "False",
        "amount_to_trade": "10",
        "max_positions": "5",
        "max_percent_per_stock": "0.2",
        "max_daily_loss": "100",
        "update_interval_sec": "60",
        "live_entry_ttl_seconds": "120",
        "db_dir": "../db",
    }
    monkeypatch.setattr(main, "load_split_config", lambda _paths: cfg)

    def _boom():
        raise RuntimeError("report read failed")

    monkeypatch.setattr(main, "get_last_config_sanitizer_report", _boom)

    class DummyDataManager:
        def __init__(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(main, "DataManager", DummyDataManager)

    with caplog.at_level(logging.ERROR):
        main.main()

    assert any("Failed to read sanitizer report" in rec.message for rec in caplog.records)
