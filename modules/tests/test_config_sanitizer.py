import os


from modules.config_io import load_split_config, get_last_config_sanitizer_report


def test_load_split_config_repairs_merged_configuration_lines(tmp_path):
    root = tmp_path / "TradingBot"
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True)

    # Intentionally corrupted: multiple key assignments on one line inside [CONFIGURATION]
    config_ini = cfg_dir / "config.ini"
    config_ini.write_text(
        "[CONFIGURATION]\n"
        "amount_to_trade=2000 max_positions=5 max_percent_per_stock=0.2\n",
        encoding="utf-8",
    )

    (cfg_dir / "keys.ini").write_text(
        "[KEYS]\n"
        "alpaca_key=\n"
        "alpaca_secret=\n",
        encoding="utf-8",
    )
    (cfg_dir / "watchlist.ini").write_text(
        "[WATCHLIST_ACTIVE_STOCK]\n",
        encoding="utf-8",
    )
    (cfg_dir / "strategy.ini").write_text(
        "[STRATEGY_THE_GENERAL]\n"
        "rsi_buy=35\n",
        encoding="utf-8",
    )

    paths = {
        "configuration_ini": str(config_ini),
        "watchlist_ini": str(cfg_dir / "watchlist.ini"),
        "strategy_ini": str(cfg_dir / "strategy.ini"),
        "keys_ini": str(cfg_dir / "keys.ini"),
    }

    cfg = load_split_config(paths)
    rep = get_last_config_sanitizer_report()

    assert cfg.get("CONFIGURATION", "amount_to_trade") == "2000"
    assert cfg.get("CONFIGURATION", "max_positions") == "5"
    assert cfg.get("CONFIGURATION", "max_percent_per_stock") == "0.2"

    assert rep is not None
    assert rep.get("repairs", 0) >= 1
    assert rep.get("backup_path")
    assert os.path.exists(rep["backup_path"])

    # Ensure file was rewritten into one assignment per line.
    repaired = config_ini.read_text(encoding="utf-8")
    assert "amount_to_trade" in repaired
    assert "max_positions" in repaired
    assert "max_percent_per_stock" in repaired
    assert "amount_to_trade=2000" in repaired.replace(" ", "")
    assert "max_positions=5" in repaired.replace(" ", "")
    assert "max_percent_per_stock=0.2" in repaired.replace(" ", "")


def test_load_split_config_hydrates_optional_telegram_keys(tmp_path):
    root = tmp_path / "TradingBot"
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True)

    (cfg_dir / "config.ini").write_text(
        "[CONFIGURATION]\n"
        "paper_trading=True\n",
        encoding="utf-8",
    )
    (cfg_dir / "keys.ini").write_text(
        "[KEYS]\n"
        "alpaca_key=paper_key\n"
        "alpaca_secret=paper_secret\n",
        encoding="utf-8",
    )
    (cfg_dir / "watchlist.ini").write_text("[WATCHLIST_ACTIVE_STOCK]\n", encoding="utf-8")
    (cfg_dir / "strategy.ini").write_text("[STRATEGY_THE_GENERAL]\n", encoding="utf-8")

    paths = {
        "configuration_ini": str(cfg_dir / "config.ini"),
        "watchlist_ini": str(cfg_dir / "watchlist.ini"),
        "strategy_ini": str(cfg_dir / "strategy.ini"),
        "keys_ini": str(cfg_dir / "keys.ini"),
    }

    cfg = load_split_config(paths)

    assert cfg.has_option("KEYS", "telegram_token")
    assert cfg.get("KEYS", "telegram_token") == ""
    assert cfg.has_option("KEYS", "telegram_chat_id")
    assert cfg.get("KEYS", "telegram_chat_id") == ""
    assert cfg.has_option("KEYS", "telegram_enabled")
    assert cfg.get("KEYS", "telegram_enabled") == ""
