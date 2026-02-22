import builtins
import configparser
import importlib
import logging
import sys
import types

import pytest


def _base_cfg(*, strict: bool, amount_to_trade: str) -> configparser.ConfigParser:
    c = configparser.ConfigParser()
    c["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}
    c["CONFIGURATION"] = {
        "strict_config_validation": "True" if strict else "False",
        "amount_to_trade": amount_to_trade,
        "max_positions": "5",
        "max_percent_per_stock": "0.2",
        "max_daily_loss": "100",
        "update_interval_sec": "60",
        "live_entry_ttl_seconds": "120",
        "db_dir": "../db",
    }
    return c


def _bootstrap_main_with_dummies(monkeypatch, tmp_path):
    """Patch heavyweight startup dependencies and return (main, app_created dict)."""
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

    main = importlib.import_module("modules.main")
    monkeypatch.setattr(main, "TradingApp", DummyTradingApp)

    logs = tmp_path / "logs"
    backup = tmp_path / "backup"
    root = tmp_path / "root"

    monkeypatch.setattr(
        main,
        "get_paths",
        lambda: {
            "logs": str(logs),
            "backup": str(backup),
            "root": str(root),
            "config_dir": str(root / "config"),
            "db_dir": str(root / "db"),
        },
    )
    monkeypatch.setattr(main, "ensure_split_config_layout", lambda _paths: None)

    class DummyDataManager:
        def __init__(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(main, "DataManager", DummyDataManager)
    return main, app_created


def test_main_raises_on_strict_validation_errors(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=True, amount_to_trade="0"))
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Configuration validation failed in strict mode"):
            main.main()

    assert app_created["value"] is False
    assert any("amount_to_trade must be > 0" in rec.message for rec in caplog.records)


def test_main_continues_on_non_strict_validation_errors(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=False, amount_to_trade="0"))
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    with caplog.at_level(logging.ERROR):
        main.main()

    assert app_created["value"] is True
    assert any("amount_to_trade must be > 0" in rec.message for rec in caplog.records)


def test_main_logs_when_sanitizer_report_access_fails(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=False, amount_to_trade="10"))

    def _boom():
        raise RuntimeError("report read failed")

    monkeypatch.setattr(main, "get_last_config_sanitizer_report", _boom)

    with caplog.at_level(logging.ERROR):
        main.main()

    assert app_created["value"] is True
    assert any("Failed to read sanitizer report" in rec.message for rec in caplog.records)


def test_main_logs_when_sanitizer_log_write_fails(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=False, amount_to_trade="10"))
    monkeypatch.setattr(
        main,
        "get_last_config_sanitizer_report",
        lambda: {"repairs": 1, "backup_path": "backup.ini"},
    )

    real_open = builtins.open

    def _failing_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("config_sanitizer.log") and "a" in mode:
            raise OSError("disk full")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _failing_open)

    with caplog.at_level(logging.ERROR):
        main.main()

    assert app_created["value"] is True
    assert any("Failed writing config_sanitizer.log" in rec.message for rec in caplog.records)

def test_main_boots_without_broker_credentials(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=False, amount_to_trade="100"))
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    with caplog.at_level(logging.WARNING):
        main.main()

    assert app_created["value"] is True
    assert any("KEYS.alpaca_key is empty" in rec.message for rec in caplog.records)
    assert any("KEYS.alpaca_secret is empty" in rec.message for rec in caplog.records)


def test_main_raises_on_strict_missing_credentials(monkeypatch, tmp_path, caplog):
    main, app_created = _bootstrap_main_with_dummies(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "load_split_config", lambda _paths: _base_cfg(strict=True, amount_to_trade="100"))
    monkeypatch.setattr(main, "get_last_config_sanitizer_report", lambda: None)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Configuration validation failed in strict mode"):
            main.main()

    assert app_created["value"] is False
    assert any("KEYS.alpaca_key is empty" in rec.message for rec in caplog.records)
