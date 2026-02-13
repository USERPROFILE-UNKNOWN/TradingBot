import configparser

from modules.config_validate import validate_runtime_config


def _base_cfg():
    c = configparser.ConfigParser()
    c["KEYS"] = {"alpaca_key": "", "alpaca_secret": ""}
    c["CONFIGURATION"] = {
        "amount_to_trade": "2000",
        "max_positions": "5",
        "update_interval_sec": "60",
        "max_percent_per_stock": "0.2",
        "max_daily_loss": "100",
        "live_entry_ttl_seconds": "120",
    }
    return c


def test_validate_runtime_config_happy_path_has_no_errors():
    cfg = _base_cfg()
    rep = validate_runtime_config(cfg)
    assert rep.ok is True
    assert rep.errors == []
    assert len(rep.warnings) >= 1


def test_validate_runtime_config_rejects_invalid_ranges():
    cfg = _base_cfg()
    cfg["CONFIGURATION"]["amount_to_trade"] = "0"
    cfg["CONFIGURATION"]["max_positions"] = "0"
    cfg["CONFIGURATION"]["max_percent_per_stock"] = "1.5"
    cfg["CONFIGURATION"]["max_daily_loss"] = "0"
    cfg["CONFIGURATION"]["live_entry_ttl_seconds"] = "-5"

    rep = validate_runtime_config(cfg)
    assert rep.ok is False
    assert any("amount_to_trade" in e for e in rep.errors)
    assert any("max_positions" in e for e in rep.errors)
    assert any("max_percent_per_stock" in e for e in rep.errors)
    assert any("max_daily_loss" in e for e in rep.errors)
    assert any("live_entry_ttl_seconds" in e for e in rep.errors)


def test_validate_runtime_config_warns_on_very_low_update_interval():
    cfg = _base_cfg()
    cfg["CONFIGURATION"]["update_interval_sec"] = "2"
    rep = validate_runtime_config(cfg)
    assert any("update_interval_sec" in w for w in rep.warnings)
