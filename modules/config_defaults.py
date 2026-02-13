"""Default split-config builder (v5.11.0 defaults to split DB mode).

This module owns the default values for the 4-file split INI layout.

It is intentionally isolated from UI/engine code to reduce regression risk.
"""

from __future__ import annotations

import configparser


def _new_config_parser() -> configparser.ConfigParser:
    """Create a robust ConfigParser.

    - Preserve option name case (optionxform=str)
    - Disable interpolation to avoid '%' parsing errors
    - Allow inline comments (# / ;)
    - Allow duplicates in damaged INI files (last one wins)
    """
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        strict=False,
    )
    cfg.optionxform = str
    return cfg



def default_split_config() -> configparser.ConfigParser:
    """Return a ConfigParser containing the default split-config layout."""
    cfg = _new_config_parser()

    cfg["KEYS"] = {
        "base_url": "https://paper-api.alpaca.markets",
        "alpaca_key": "",
        "alpaca_secret": "",
        "telegram_token": "",
        "telegram_chat_id": "",
        "telegram_enabled": "False",
    }

    cfg["CONFIGURATION"] = {
        "max_positions": "5",
        "amount_to_trade": "2000",
        "max_percent_per_stock": "0.20",
        "max_daily_loss": "100",
        "compounding_enabled": "True",
        "ai_sizing_enabled": "True",
        "agent_mode": "OFF",
        "agent_live_max_exposure_pct": "0.30",
        "agent_max_live_changes_per_day": "8",
        "update_interval_sec": "60",
"db_dir": "..\\db",
# Logging / observability knobs
        "log_level": "INFO",
        "log_snapshot_interval_sec": "300",
        "log_scan_summary": "True",
        "log_candidate_lines": "False",
# Candidate Scanner (v5.13.1 Update A)
"candidate_scanner_enabled": "False",
"candidate_scanner_universe_mode": "WATCHLIST",    # WATCHLIST | ALL_DB_SYMBOLS | CONFIG_LIST
"candidate_scanner_universe": "",                  # comma-separated symbols when universe_mode=CONFIG_LIST
"candidate_scanner_limit": "20",
"candidate_scanner_lookback_bars": "390",
"candidate_scanner_min_bars": "120",
"candidate_scanner_min_dollar_volume": "1000000",
"candidate_scanner_score_mode": "simple",
"candidate_scanner_include_negative_movers": "True",

# Dynamic Watchlist Policy (v5.13.1 Update B)
"watchlist_auto_update_enabled": "False",
"watchlist_auto_update_interval_min": "30",
"watchlist_auto_update_mode": "ADD",              # ADD | REPLACE
"watchlist_auto_update_max_add": "5",
"watchlist_auto_update_max_total": "20",
"watchlist_auto_update_min_score": "0.0",

# Crypto Stable Set (v5.13.1 Update B)
"crypto_stable_set_enabled": "False",
"crypto_stable_set_replace_existing": "True",
"crypto_stable_set_lookback_bars": "390",
"crypto_stable_set_min_dollar_volume": "5000000",
"crypto_stable_set_max_spread_pct": "0.5",
"crypto_stable_set_max_assets": "6",

        "log_decision_rejects": "False",
        "log_guardian_banner": "False",

        # UI/telemetry log controls
        "log_file_enabled": "True",
        "log_file_prefix": "RUNTIME",
        "log_file_roll_daily": "True",
        "log_max_lines": "2000",

        # v5.13.2 updateA: Mini-player (Live Log)
        # When enabled, UI refresh slows down and heavy redraw loops (charts/heatmaps) pause.
        "miniplayer_enabled": "False",
        "ui_refresh_ms": "2000",
        "miniplayer_ui_refresh_ms": "8000",

        # Config tab / configuration management (v5.12.7)
        "config_auto_update_enabled": "False",
        "config_history_enabled": "True",
        "config_history_max_rows": "5000",

        # AI retraining + run summary + data freshness
        "ai_retrain_after_db_update": "True",
        "run_summary_enabled": "True",
        "run_summary_prefix": "SUMMARY",
        "stale_bar_seconds_threshold": "180",
        "stale_bar_log_each_symbol": "False",

        # Backtest realism / fills
        "backtest_entry_ttl_bars": "2",
        "backtest_bar_conflict": "STOP_FIRST",
        "backtest_slippage_bps": "5",
        "backtest_fee_per_trade": "0",
        "backtest_atr_stop_mult": "2.0",
        "backtest_atr_take_mult": "3.0",
        "backtest_max_hold_bars": "240",
        "backtest_use_stagnation_exit": "True",
        "backtest_stagnation_minutes": "60",
        "backtest_stagnation_min_gain": "-0.01",
        "backtest_stagnation_max_gain": "0.003",

        # Engine guardrails
        "guardian_sell_on_negative_sentiment": "False",
        "guardian_sentiment_threshold": "-0.3",
        "guardian_use_stagnation_exit": "True",
        "guardian_stagnation_minutes": "60",
        "guardian_stagnation_min_gain": "-0.01",
        "guardian_stagnation_max_gain": "0.003",

        # Confirmation logic
        "confirmation_required_hits": "2",
        "confirmation_ttl_minutes": "10",
        "confirmation_min_delay_seconds": "60",

        # Order lifecycle
        "log_order_lifecycle": "True",
        "live_entry_ttl_seconds": "120",
        "live_order_poll_interval_seconds": "5",
        "live_cancel_unfilled_entries": "True",

        # Architect alignment
        "architect_entry_mode": "LIMIT_CLOSE_TTL",
        "architect_entry_ttl_bars": "2",
        "architect_use_trailing_stop": "True",
        "architect_trailing_uses_sl": "True",
        "architect_use_stagnation_exit": "True",
        "architect_stagnation_bars": "60",
        "architect_stagnation_min_gain": "-0.01",
        "architect_stagnation_max_gain": "0.003",
        "architect_use_atr_exits": "False",
        "architect_atr_stop_mult": "2.0",
        "architect_atr_take_mult": "3.0",
        "architect_max_hold_bars": "200",
        "architect_min_trades": "5",
        "architect_use_walkforward": "False",
        "architect_walkforward_splits": "3",
        "architect_bar_conflict": "STOP_FIRST",
    }

    cfg["WATCHLIST_FAVORITES_STOCK"] = {
        # Add your favorites (stocks) here
    }

    cfg["WATCHLIST_FAVORITES_CRYPTO"] = {
        # Add your favorites (crypto) here
    }

    cfg["WATCHLIST_ACTIVE_STOCK"] = {
        "TQQQ": "",
        "SOXL": "",
        "AMD": "",
        "PLTR": "",
        "UBER": "",
        "HOOD": "",
        "MARA": "",
        "COIN": "",
        "XLE": "",
        "SQQQ": "",
        "SOXS": "",
        "SPXU": "",
    }

    cfg["WATCHLIST_ACTIVE_CRYPTO"] = {
        "BTC/USD": "",
        "ETH/USD": "",
    }

    cfg["WATCHLIST_ARCHIVE_STOCK"] = {
        # Archive stocks here
    }

    cfg["WATCHLIST_ARCHIVE_CRYPTO"] = {
        # Archive crypto here
    }

    cfg["STRATEGY_THE_GENERAL"] = {
        "rsi_buy": "35",
        "bollinger_std": "2.0",
        "ema_period": "200",
        "adx_min": "20",
        "require_confirmation": "True",
        "trailing_stop_pct": "0.02",
        "stop_loss": "0.02",
        "min_rvol": "1.2",
    }

    cfg["STRATEGY_SNIPER"] = {
        "rsi_buy": "25",
        "bollinger_std": "2.5",
        "ema_period": "200",
        "adx_min": "25",
        "require_confirmation": "True",
        "trailing_stop_pct": "0.015",
        "stop_loss": "0.015",
        "min_rvol": "2.0",
    }

    cfg["STRATEGY_MOMENTUM"] = {
        "rsi_buy": "45",
        "bollinger_std": "1.5",
        "ema_period": "50",
        "adx_min": "30",
        "require_confirmation": "False",
        "trailing_stop_pct": "0.04",
        "stop_loss": "0.05",
        "min_rvol": "1.0",
    }

    cfg["STRATEGY_BREAKOUT"] = {
        "rsi_buy": "60",
        "bollinger_std": "2.0",
        "ema_period": "50",
        "adx_min": "25",
        "require_confirmation": "False",
        "trailing_stop_pct": "0.03",
        "stop_loss": "0.03",
        "min_rvol": "2.5",
    }

    return cfg


# Backward-compatible alias (old code used _default_split_config in utils.py)
_default_split_config = default_split_config
