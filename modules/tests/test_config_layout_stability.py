from modules.config_io import ensure_split_config_layout


def test_ensure_split_config_layout_keeps_existing_config_ini_unchanged(tmp_path):
    root = tmp_path / "TradingBot"
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True)

    config_ini = cfg_dir / "config.ini"
    original = (
        "[CONFIGURATION]\n"
        "strict_config_validation = False\n"
        "\n"
        "#  ==========================\n"
        "# TRADINGVIEW (Webhook Integration)\n"
        "#  ==========================\n"
        "enabled = False\n"
    )
    config_ini.write_text(original, encoding="utf-8")

    (cfg_dir / "keys.ini").write_text("[KEYS]\n", encoding="utf-8")
    (cfg_dir / "watchlist.ini").write_text("[WATCHLIST_ACTIVE_STOCK]\n", encoding="utf-8")
    (cfg_dir / "strategy.ini").write_text("[STRATEGY_THE_GENERAL]\nrsi_buy = 35\n", encoding="utf-8")

    paths = {
        "config_dir": str(cfg_dir),
        "configuration_ini": str(config_ini),
        "keys_ini": str(cfg_dir / "keys.ini"),
        "watchlist_ini": str(cfg_dir / "watchlist.ini"),
        "strategy_ini": str(cfg_dir / "strategy.ini"),
    }

    before = config_ini.read_text(encoding="utf-8")
    ensure_split_config_layout(paths)
    after = config_ini.read_text(encoding="utf-8")

    assert after == before
    assert "CONFIGURATION DEFAULTS SYNC" not in after
