"""Canonical split-config defaults (v6.15.0).

This module is the single source of truth for default values used to:
- Create missing config/*.ini files (fresh install)
- Repair missing keys/sections without rewriting user values

Rules:
- Secrets live in config/keys.ini (never auto-added to config/config.ini).
- Forward-only: no legacy/compat shims; remove obsolete keys during repair when safe.
"""

from __future__ import annotations

import configparser
from typing import Dict

# ---- Default key/value schema (string values; ConfigParser stores everything as text) ----
CONFIGURATION_DEFAULTS: Dict[str, str] = {
    "agent_live_max_exposure_pct": "0.30",
    "agent_max_live_changes_per_day": "8",
    "agent_mode": "OFF",
    "agent_auto_backfill_enabled": "True",
    "agent_candidate_scan_enabled": "True",
    "agent_candidate_scan_interval_minutes": "60",
    "agent_candidate_simulation_enabled": "True",
    "agent_candidate_simulation_interval_minutes": "60",
    "agent_candidate_simulation_max_symbols": "10",
    "agent_candidate_simulation_run_after_scan": "True",
    "agent_watchlist_policy_enabled": "True",
    "agent_watchlist_policy_interval_minutes": "60",
    "agent_watchlist_policy_max_churn_per_run": "10",
    "agent_watchlist_policy_run_after_simulation": "True",
    "agent_quick_backtest_enabled": "True",
    "agent_quick_backtest_interval_minutes": "1440",
    "agent_quick_backtest_max_symbols": "10",
    "agent_quick_backtest_days": "14",
    "agent_quick_backtest_max_strategies": "6",
    "agent_quick_backtest_min_trades": "1",
    "agent_architect_optimize_enabled": "True",
    "agent_architect_optimize_interval_minutes": "10080",
    "agent_architect_optimize_max_symbols": "5",
    "agent_architect_optimize_top_variants_per_symbol": "3",
    "agent_architect_orchestrator_enabled": "True",
    "agent_architect_orchestrator_interval_minutes": "10080",
    "agent_architect_orchestrator_hour_utc": "3",
    "agent_architect_orchestrator_max_queue_items": "10",
    "agent_architect_orchestrator_max_symbols": "25",
    "agent_full_backtest_enabled": "False",
    "agent_full_backtest_interval_minutes": "1440",
    "agent_full_backtest_hour_utc": "2",
    "agent_backfill_min_interval_minutes": "240",
    "agent_daily_report_enabled": "True",
    "agent_daily_report_hour_utc": "0",
    "agent_db_integrity_check_enabled": "True",
    "agent_stale_quarantine_enabled": "True",
    "agent_stale_quarantine_threshold_seconds": "21600",
    "agent_stale_quarantine_warmup_minutes": "45",
    "agent_research_automation_enabled": "True",
    "agent_research_sweep_hour_utc": "1",
    "agent_research_max_symbols": "10",
    "agent_research_backtest_days": "14",
    "agent_research_max_strategies": "6",
    "agent_research_min_trades": "1",
    "agent_canary_enabled": "True",
    "agent_canary_exposure_pct": "0.10",
    "agent_canary_reject_rate_pct_max": "10.0",
    "agent_canary_slippage_bps_max": "25.0",
    "agent_canary_drawdown_pct_max": "4.0",
    "agent_canary_underperform_pct_max": "5.0",
    "agent_canary_rollback_mode": "PAPER",
    "agent_hard_halt_supreme": "True",
    "agent_config_tuning_enabled": "True",
    "agent_max_config_tunes_per_day": "2",
    "agent_promotion_enabled": "True",
    "agent_max_promotions_per_day": "1",
    "paper_trading": "True",
    "ai_calibration_frac": "0.2",
    "ai_calibration_fraction": "0.2",
    "ai_calibration_method": "sigmoid",
    "ai_label_horizon_bars": "60",
    "ai_label_min_return_pct": "0.001",
    "ai_label_sl_atr_mult": "1.0",
    "ai_label_tp_atr_mult": "1.5",
    "ai_min_prob": "0.50",
    "ai_min_train_rows": "2000",
    "ai_retrain_after_db_update": "True",
    "ai_rf_estimators": "200",
    "ai_rf_min_leaf": "5",
    "ai_sizing_enabled": "True",
    "ai_train_limit_per_symbol": "20000",
    "ai_walkforward_splits": "3",
    "amount_to_trade": "2000",
    "architect_atr_stop_mult": "2.0",
    "architect_atr_take_mult": "3.0",
    "architect_bar_conflict": "STOP_FIRST",
    "architect_entry_mode": "LIMIT_CLOSE_TTL",
    "architect_entry_ttl_bars": "2",
    "architect_max_hold_bars": "200",
    "architect_min_trades": "5",
    "architect_stagnation_bars": "60",
    "architect_stagnation_max_gain": "0.003",
    "architect_stagnation_min_gain": "-0.01",
    "architect_trailing_stop_pct": "0.02",
    "architect_trailing_uses_sl": "True",
    "architect_use_atr_exits": "False",
    "architect_use_stagnation_exit": "True",
    "architect_use_trailing_stop": "True",
    "architect_use_walkforward": "False",
    "architect_walkforward_splits": "3",
    "backtest_atr_stop_mult": "2.0",
    "backtest_atr_take_mult": "3.0",
    "backtest_bar_conflict": "STOP_FIRST",
    "backtest_entry_ttl_bars": "2",
    "backtest_fee_per_trade": "0",
    "backtest_max_hold_bars": "240",
    "backtest_slippage_bps": "5",
    "backtest_stagnation_max_gain": "0.003",
    "backtest_stagnation_min_gain": "-0.01",
    "backtest_stagnation_minutes": "60",
    "backtest_use_stagnation_exit": "True",
    "candidate_scanner_enabled": "True",
    "candidate_scanner_max_symbols": "50",
    "candidate_scanner_min_liquidity_usd": "500000",
    "candidate_scanner_max_spread_bps": "30",
    "candidate_scanner_min_price": "2.0",
    "candidate_scanner_lookback_days": "90",
    "candidate_scanner_refresh_minutes": "60",
    "candidate_scanner_include_negative_movers": "True",
    "candidate_scanner_limit": "20",
    "candidate_scanner_lookback_bars": "390",
    "candidate_scanner_min_bars": "120",
    "candidate_scanner_min_dollar_volume": "1000000",
    "candidate_scanner_score_mode": "simple",
    "candidate_scanner_universe": "",
    "candidate_scanner_universe_mode": "WATCHLIST",
    "compounding_enabled": "True",
    "config_auto_update_enabled": "True",
    "config_history_enabled": "True",
    "config_history_max_rows": "5000",
    "config_history_max_snapshots": "25",
    "config_history_rotate_on_change": "True",
    "confirmation_min_delay_seconds": "60",
    "confirmation_required_hits": "2",
    "confirmation_ttl_minutes": "10",
    "crypto_stable_set_enabled": "False",
    "crypto_stable_set": "BTC/USD,ETH/USD,SOL/USD",
    "crypto_stable_set_lookback_bars": "390",
    "crypto_stable_set_max_assets": "6",
    "crypto_stable_set_max_spread_pct": "0.5",
    "crypto_stable_set_min_dollar_volume": "5000000",
    "crypto_stable_set_replace_existing": "True",
    "daily_budget": "1000",
    "db_auto_migrate_on_startup": "True",
    "db_dir": "db",
    "db_mode": "SPLIT",
    "db_path": "..\\db\\market_data.db",
    "e4_correlation_clusters": "TECH_LEVERAGED:TQQQ,SOXL|INVERSE_LEVERAGED:SQQQ,SOXS,SPXU|CRYPTO:BTC/USD,ETH/USD|CRYPTO_MINERS:MARA,COIN|FINTECH:HOOD,COIN",
    "e4_enabled": "True",
    "e4_max_positions_per_cluster": "1",
    "e4_mtf_ema_fast": "50",
    "e4_mtf_ema_slow": "200",
    "e4_mtf_enabled": "True",
    "e4_mtf_refresh_seconds": "120",
    "e4_mtf_rsi_min_momentum": "50",
    "e4_mtf_timeframes": "5Min,15Min",
    "e5_enabled": "True",
    "e5_enforce_on_live": "True",
    "e5_enforce_on_paper": "False",
    "e5_event_window_sec": "900",
    "e5_halt_cancel_pending_orders": "True",
    "e5_halt_liquidate": "False",
    "e5_halt_stop_engine": "False",
    "e5_max_api_errors_in_window": "20",
    "e5_max_rejects_in_window": "10",
    "e5_max_ttl_cancels_in_window": "20",
    "e5_watchdog_interval_sec": "30",
    "equity_data_feed": "iex",
    "guardian_sell_on_negative_sentiment": "False",
    "guardian_sentiment_threshold": "-0.3",
    "guardian_stagnation_max_gain": "0.003",
    "guardian_stagnation_min_gain": "-0.01",
    "guardian_stagnation_minutes": "60",
    "guardian_use_stagnation_exit": "True",
    "indicator_warmup_minutes": "500",
    "live_cancel_unfilled_entries": "True",
    "live_entry_ttl_seconds": "120",
    "live_order_poll_interval_seconds": "5",
    "log_candidate_lines": "False",
    "log_decision_rejects": "False",
    "log_file_enabled": "True",
    "log_file_prefix": "RUNTIME",
    "log_file_roll_daily": "True",
    "log_guardian_banner": "False",
    "log_level": "INFO",
    "log_max_lines": "2000",
    "log_order_lifecycle": "True",
    "log_scan_summary": "True",
    "log_snapshot_interval_sec": "300",
    "market_regime_symbol": "SPY",
    "max_daily_loss": "100",
    "max_open_trades": "5",
    "max_percent_per_stock": "0.20",
    "max_positions": "5",
    "max_qty": "100",
    "min_equity": "1000",
    "miniplayer_enabled": "True",
    "miniplayer_ui_refresh_ms": "8000",
    "miniplayer_update_interval_ms": "250",
    "miniplayer_max_rows": "12",
    "promotion_enabled": "True",
    "promotion_live_scale_days": "5",
    "promotion_live_start_fraction": "0.25",
    "promotion_max_api_errors_per_hour": "20",
    "promotion_max_cancel_rate_pct": "35.0",
    "promotion_max_daily_loss_pct": "2.0",
    "promotion_max_drawdown_pct": "4.0",
    "promotion_max_reject_rate_pct": "10.0",
    "promotion_min_sessions": "5",
    "promotion_min_trades_per_symbol": "3",
    "promotion_min_trades_total": "30",
    "promotion_require_no_crashes": "True",
    "promotion_require_no_stale_symbols": "True",
    "promotion_require_no_watchdog_halts": "True",
    "promotion_stale_threshold_seconds": "180",
    "promotion_window_days": "30",
    "reconcile_confirmations": "2",
    "reconcile_halt_on_mismatch": "True",
    "reconcile_interval_seconds": "60",
    "regime_chop_adx_max": "18.0",
    "regime_highvol_atr_pct_min": "0.03",
    "risk_per_trade_pct": "1.0",
    "run_summary_enabled": "True",
    "run_summary_prefix": "SUMMARY",
    "stale_bar_log_each_symbol": "False",
    "stale_bar_seconds_threshold": "180",
    "strict_config_validation": "False",
    "enabled": "False",
    "listen_host": "127.0.0.1",
    "listen_port": "5001",
    "secret": "",
    "allowed_signals": "",
    "mode": "ADVISORY",
    "candidate_cooldown_minutes": "5",
    "autovalidation_enabled": "True",
    "autovalidation_cooldown_minutes": "10",
    "autovalidation_freshness_minutes": "30",
    "autovalidation_backfill_days": "60",
    "autovalidation_backtest_days": "14",
    "autovalidation_max_strategies": "6",
    "autovalidation_min_trades": "1",
    "autovalidation_max_concurrency": "1",
    "ui_refresh_ms": "2000",
    "update_db_lookback_days": "30",
    "update_interval_sec": "60",
    "watchlist_auto_update_enabled": "True",
    "watchlist_auto_update_minutes": "60",
    "watchlist_auto_update_interval_min": "30",
    "watchlist_auto_update_max_add": "5",
    "watchlist_auto_update_max_total": "20",
    "watchlist_auto_update_min_score": "0.0",
    "watchlist_auto_update_mode": "ADD",
}


KEYS_DEFAULTS: Dict[str, str] = {
    "paper_base_url": "https://paper-api.alpaca.markets",
    "paper_alpaca_key": "",
    "paper_alpaca_secret": "",
    "live_base_url": "https://api.alpaca.markets",
    "live_alpaca_key": "",
    "live_alpaca_secret": "",
    "telegram_token": "",
    "telegram_chat_id": "",
    "telegram_enabled": "True",
}

WATCHLIST_INI_TEMPLATE = r"""
[WATCHLIST_FAVORITES_STOCK]

[WATCHLIST_FAVORITES_CRYPTO]

[WATCHLIST_FAVORITES_ETF]

[WATCHLIST_ACTIVE_STOCK]
TQQQ = 
SOXL = 
AMD = 
PLTR = 
UBER = 
HOOD = 
MARA = 
COIN = 
XLE = 
SQQQ = 
SOXS = 
SPXU = 

[WATCHLIST_ACTIVE_CRYPTO]
AAVE/USD = 
AVAX/USD = 
BAT/USD = 
BCH/USD = 
BTC/USD = 
CRV/USD = 
DOGE/USD = 
DOT/USD = 
ETH/USD = 
GRT/USD = 
LINK/USD = 
LTC/USD = 
PEPE/USD = 
SHIB/USD = 
SKY/USD = 
SOL/USD = 
SUSHI/USD = 
TRUMP/USD = 
UNI/USD = 
USDC/USD = 
USDG/USD = 
USDT/USD = 
XRP/USD = 
XTZ/USD = 
YFI/USD = 

[WATCHLIST_ACTIVE_ETF]
SOXL = 
SOXS = 
SPXU = 
SQQQ = 
TQQQ = 

[WATCHLIST_ARCHIVE_STOCK]
TQQQ = 
SOXL = 
AMD = 
PLTR = 
UBER = 
HOOD = 
MARA = 
COIN = 
XLE = 
SQQQ = 
SOXS = 
SPXU = 

[WATCHLIST_ARCHIVE_CRYPTO]
AAVE/USD = 
AVAX/USD = 
BAT/USD = 
BCH/USD = 
BTC/USD = 
CRV/USD = 
DOGE/USD = 
DOT/USD = 
ETH/USD = 
GRT/USD = 
LINK/USD = 
LTC/USD = 
PEPE/USD = 
SHIB/USD = 
SKY/USD = 
SOL/USD = 
SUSHI/USD = 
TRUMP/USD = 
UNI/USD = 
USDC/USD = 
USDG/USD = 
USDT/USD = 
XRP/USD = 
XTZ/USD = 
YFI/USD = 

[WATCHLIST_ARCHIVE_ETF]
SOXL = 
SOXS = 
SPXU = 
SQQQ = 
TQQQ = 
"""

SECTORS_INI_TEMPLATE = r"""
[STOCK_SECTORS]
Commercial services = 
Communications = 
Consumer durables = 
Consumer non-durables = 
Consumer services = 
Distribution services = 
Electronic technology = 
Energy minerals = 
Finance = 
Government = 
Health services = 
Health technology = 
Industrial services = 
Miscellaneous = 
Non-energy minerals = 
Process industries = 
Producer manufacturing = 
Retail trade = 
Technology services = 
Transportation = 
Utilities = 

[CRYPTO_SECTORS]
Algorithmic stablecoins = 
Analytics = 
Animal memes = 
Asset management = 
Asset-backed stablecoins = 
Asset-backed tokens = 
Centralized exchange = 
Cryptocurrencies = 
Cybersecurity = 
DAO = 
Data management and AI = 
Decentralized exchange = 
DeFi = 
DePIN = 
Derivatives = 
Development tools = 
Distributed computing and storage = 
E-commerce = 
Education = 
Energy = 
Enterprise solutions = 
Events = 
Exchange tokens = 
Fan tokens = 
Fiat-backed stablecoins = 
Fundraising = 
Gambling = 
Gaming = 
Health = 
Hospitality = 
Identity = 
Insurance = 
Internet of things = 
Interoperability = 
ISO 20022 = 
Jobs = 
Layer 1 = 
Lending and borrowing = 
Logistics = 
Loyalty and rewards = 
Made in America = 
Made in China = 
Manufacturing = 
Marketing = 
Marketplace = 
Memes = 
Metaverse = 
Move to earn = 
NFTs and collectables = 
Oracles = 
Payments = 
Prediction markets = 
Privacy = 
Real estate = 
Real-world assets = 
Rehypothecated assets = 
Scaling = 
Seigniorage = 
Smart contract platforms = 
Social media and content = 
Sports = 
Stablecoins = 
Tap to earn = 
Tourism = 
Transport = 
Web3 = 
World Liberty Financial portfolio = 
Wrapped tokens = 

[ETF_SECTORS]
Agriculture = 
Asset allocation = 
Basket = 
Broad market = 
Broad market, asset-backed = 
Broad market, broad-based = 
Corporate, asset-backed = 
Corporate, bank loans = 
Corporate, broad-based = 
Corporate, convertible = 
Corporate, preferred = 
Energy = 
Government, agency = 
Government, broad-based = 
Government, inflation-linked = 
Government, local authority/municipal = 
Government, mortgage-backed = 
Government, non-native currency = 
Government, treasury = 
Hedge fund strategies = 
High dividend yield = 
Industrial metals = 
Metals = 
Pair = 
Precious metals = 
Sector = 
Size and style = 
Structured outcome = 
Tactical tools = 
"""

STRATEGY_INI_TEMPLATE = r"""
[STRATEGY_THE_GENERAL]
rsi_buy = 35
bollinger_std = 2.0
ema_period = 200
adx_min = 20
require_confirmation = True
trailing_stop_pct = 0.02
stop_loss = 0.02
min_rvol = 1.2

[STRATEGY_SNIPER]
rsi_buy = 25
bollinger_std = 2.5
ema_period = 200
adx_min = 25
require_confirmation = True
trailing_stop_pct = 0.015
stop_loss = 0.015
min_rvol = 2.0

[STRATEGY_MOMENTUM]
rsi_buy = 45
bollinger_std = 1.5
ema_period = 50
adx_min = 30
require_confirmation = False
trailing_stop_pct = 0.04
stop_loss = 0.05
min_rvol = 1.0

[STRATEGY_BREAKOUT]
rsi_buy = 60
bollinger_std = 2.0
ema_period = 50
adx_min = 25
require_confirmation = False
trailing_stop_pct = 0.03
stop_loss = 0.03
min_rvol = 2.5

[STRATEGY_TREND_RIDER]
enabled = True
priority = 90
regimes = BULL, BEAR, HIGH_VOL
rsi_min = 52
ema_fast = 50
ema_mid = 200
adx_min = 20
min_rvol = 1.0
trailing_stop_pct = 0.03
stop_loss = 0.03

[STRATEGY_PULLBACK_BUY]
enabled = True
priority = 80
regimes = BULL
ema_trend = 200
ema_pullback = 20
rsi_max = 45
adx_min = 18
min_rvol = 1.0
trailing_stop_pct = 0.025
stop_loss = 0.02

[STRATEGY_DONCHIAN_BREAKOUT]
enabled = True
priority = 85
regimes = BULL, BEAR, HIGH_VOL
donchian_len = 20
rsi_min = 55
adx_min = 20
min_rvol = 1.2
trailing_stop_pct = 0.035
stop_loss = 0.03

[STRATEGY_ATR_BREAKOUT]
enabled = True
priority = 82
regimes = HIGH_VOL, BULL, BEAR
lookback_len = 20
atr_mult = 0.8
rsi_min = 50
adx_min = 18
min_rvol = 1.0
trailing_stop_pct = 0.04
stop_loss = 0.035

[STRATEGY_RSI2_REVERSAL]
enabled = True
priority = 75
regimes = CHOP, BULL
rsi2_buy = 5
ema_period = 200
min_rvol = 1.0
trailing_stop_pct = 0.02
stop_loss = 0.015

[STRATEGY_Z_MEAN_REVERT]
enabled = True
priority = 78
regimes = CHOP
z_entry = -2.0
z_exit = -0.5
ema_period = 200
min_rvol = 1.0
trailing_stop_pct = 0.02
stop_loss = 0.015

[STRATEGY_VWAP_REVERT]
enabled = False
priority = 70
regimes = CHOP
vwap_window = 390
vwap_dev_pct = 0.8
ema_period = 200
rsi_buy = 35
min_rvol = 1.0
trailing_stop_pct = 0.02
stop_loss = 0.015
"""

def _new_cfg() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.optionxform = str  # preserve case
    return cfg

def _ensure_section(cfg: configparser.ConfigParser, section: str, defaults: Dict[str, str]) -> None:
    if not cfg.has_section(section):
        cfg.add_section(section)
    for k, v in defaults.items():
        if not cfg.has_option(section, k):
            cfg.set(section, k, str(v))

def default_split_config() -> configparser.ConfigParser:
    """Return a ConfigParser representing the canonical v5.16.2 split-config schema."""
    cfg = _new_cfg()

    # Core
    _ensure_section(cfg, "CONFIGURATION", CONFIGURATION_DEFAULTS)
    _ensure_section(cfg, "KEYS", KEYS_DEFAULTS)

    # Watchlist / strategies templates (multi-section)
    try:
        cfg.read_string(WATCHLIST_INI_TEMPLATE)
    except Exception:
        pass
    try:
        cfg.read_string(STRATEGY_INI_TEMPLATE)
    except Exception:
        pass
    try:
        cfg.read_string(SECTORS_INI_TEMPLATE)
    except Exception:
        pass

    return cfg
