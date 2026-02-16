import pandas_ta as ta
import pandas as pd


class StrategyOptimizer:
    """Strategy selection + scoring.

    This module is intentionally *contract-stable* with the engine/UI:
    - Engine imports: StrategyOptimizer, WalletManager
    - Engine calls:
        * load_strategies()
        * choose_strategy(symbol, bars_df, market_regime=..., is_crypto=...)
        * score_opportunity(symbol, current_bar_data, df_history, strat_name)
        * requires_confirmation(strat_name)
        * calculate_exit_prices(entry_price, atr, strat_name)

    Strategy naming:
    - INI sections are named like [STRATEGY_THE_GENERAL]
    - Internally we key strategies by the suffix only (e.g. "THE_GENERAL")
      because the engine stores strategy names without the STRATEGY_ prefix.
    """

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.strategies = {}
        self.load_strategies()

    # --------------------
    # Loading
    # --------------------
    def load_strategies(self):
        self.strategies.clear()
        for section in self.config.sections():
            if not section.startswith("STRATEGY_"):
                continue

            name = section.replace("STRATEGY_", "", 1)
            self.strategies[name] = dict(self.config[section])

    # --------------------
    # Strategy selection
    # --------------------
    def assign_best_strategy(self, symbol):
        """Legacy behavior: use best backtest strategy if available; else fallback."""
        best = None
        try:
            best = self.db.get_best_strategy_for_symbol(symbol)
        except Exception:
            best = None

        # DB returns (best_strategy, best_profit, timestamp) in current builds
        if isinstance(best, (tuple, list)):
            best = best[0] if best else None

        if best and str(best) in self.strategies:
            return str(best)

        # Default fallbacks
        if "THE_GENERAL" in self.strategies:
            return "THE_GENERAL"
        if "BREAKOUT" in self.strategies:
            return "BREAKOUT"

        # Any available strategy
        return next(iter(self.strategies.keys()), "THE_GENERAL")

    def choose_strategy(self, symbol, df, market_regime="BULL", is_crypto=False):
        """Regime-aware selection.

        Preference order:
          1) If backtest best strategy exists and is compatible with the current regime, use it.
          2) Else pick highest priority enabled strategy that lists this regime.
          3) Else fallback to assign_best_strategy().

        NOTE: This only selects a strategy name; it does NOT change trading direction.
        """
        if not self.strategies:
            self.load_strategies()

        regime = str(market_regime or "BULL").upper().strip()

        def _enabled(conf: dict) -> bool:
            try:
                v = conf.get("enabled", "True")
                return str(v).strip().lower() not in ("0", "false", "no", "off")
            except Exception:
                return True

        def _priority(conf: dict) -> float:
            try:
                return float(conf.get("priority", 50))
            except Exception:
                return 50.0

        def _regimes(conf: dict):
            try:
                r = conf.get("regimes", "")
                if not r:
                    return None
                return {x.strip().upper() for x in str(r).split(",") if x.strip()}
            except Exception:
                return None

        # Candidate pool
        candidates = []
        for name, conf in self.strategies.items():
            if not _enabled(conf):
                continue
            rs = _regimes(conf)
            if rs is not None and regime not in rs:
                continue
            candidates.append((name, _priority(conf)))

        # Prefer best backtest strategy if compatible
        best = None
        try:
            best = self.db.get_best_strategy_for_symbol(symbol)
        except Exception:
            best = None
        if isinstance(best, (tuple, list)):
            best = best[0] if best else None
        if best:
            best = str(best)
            if any(n == best for n, _ in candidates):
                return best

        if candidates:
            candidates.sort(key=lambda t: t[1], reverse=True)
            return candidates[0][0]

        # Crypto: if you ever add a dedicated CRYPTO strategy, prefer it.
        if is_crypto and "CRYPTO" in self.strategies:
            return "CRYPTO"

        return self.assign_best_strategy(symbol)

    # --------------------
    # Opportunity scoring
    # --------------------
    def _safe_float(self, v, default=0.0):
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_conf(self, strategy_name: str) -> dict:
        return self.strategies.get(str(strategy_name), {})

    def score_opportunity(self, symbol, current_bar, df_history, strategy_name):
        """Return a positive score if the strategy conditions are met; else 0.

        Signature is aligned to engine.py:
            score_opportunity(symbol, current_bar_data, df_history, strat_name)
        """
        conf = self._get_conf(strategy_name)
        if not conf:
            # Unknown strategy: treat as legacy fallback
            strategy_name = self.assign_best_strategy(symbol)
            conf = self._get_conf(strategy_name)

        # Basic fields
        price = self._safe_float(current_bar.get("close"))
        if price <= 0:
            return 0

        rsi = self._safe_float(current_bar.get("rsi"))
        rsi2 = self._safe_float(current_bar.get("rsi2"))
        adx = self._safe_float(current_bar.get("adx"))
        atr = self._safe_float(current_bar.get("atr"))

        ema_20 = self._safe_float(current_bar.get("ema_20"))
        ema_50 = self._safe_float(current_bar.get("ema_50"))
        ema_200 = self._safe_float(current_bar.get("ema_200"))

        bb_lower = self._safe_float(current_bar.get("bb_lower"))
        bb_upper = self._safe_float(current_bar.get("bb_upper"))

        donchian_high = self._safe_float(current_bar.get("donchian_high"))
        zscore = self._safe_float(current_bar.get("zscore"))
        vwap = self._safe_float(current_bar.get("vwap"), default=price)

        # Relative volume
        min_rvol = self._safe_float(conf.get("min_rvol", 0.0), default=0.0)
        curr_vol = self._safe_float(current_bar.get("volume"), default=0.0)
        vol_avg = self._safe_float(current_bar.get("volume_avg"), default=0.0)

        if vol_avg <= 0 and isinstance(df_history, pd.DataFrame) and (not df_history.empty) and "volume" in df_history.columns:
            try:
                v = pd.to_numeric(df_history["volume"], errors="coerce")
                vol_avg = float(v.tail(20).mean()) if len(v) >= 5 else float(v.mean())
            except Exception:
                vol_avg = 0.0

        rvol = (curr_vol / vol_avg) if (vol_avg and vol_avg > 0) else 0.0
        if min_rvol and min_rvol > 0:
            if rvol <= 0 or rvol < min_rvol:
                return 0

        # Strategy-specific scoring. Keep it simple: satisfy conditions -> positive score.
        name = str(strategy_name).upper().strip()

        # --- Legacy 4 strategies (existing behavior) ---
        if name == "BREAKOUT":
            # Strong move above upper band + strength
            if price > bb_upper and rsi > 50 and price > ema_200:
                base = 60.0
                base += max(0.0, (rsi - 50.0)) * 0.5
                base += max(0.0, (adx - 20.0)) * 0.3
                base += min(20.0, max(0.0, rvol - 1.0) * 10.0)
                return base
            return 0

        if name in ("THE_GENERAL", "SNIPER", "MOMENTUM"):
            buy_rsi = self._safe_float(conf.get("rsi_buy", 35.0), default=35.0)
            adx_min = self._safe_float(conf.get("adx_min", 0.0), default=0.0)
            if adx_min and adx < adx_min:
                return 0

            if name == "MOMENTUM":
                # Momentum: above EMA and improving RSI
                ema_period = self._safe_float(conf.get("ema_period", 50.0), default=50.0)
                ema_ref = ema_50 if int(ema_period) <= 50 else ema_200
                if price > ema_ref and rsi >= buy_rsi:
                    base = 55.0
                    base += max(0.0, (rsi - buy_rsi)) * 0.6
                    base += max(0.0, (adx - adx_min)) * 0.3
                    base += min(15.0, max(0.0, rvol - 1.0) * 8.0)
                    return base
                return 0

            # Mean-reversion entry
            if price < bb_lower and rsi < buy_rsi and price > ema_200:
                base = 50.0
                base += max(0.0, (buy_rsi - rsi)) * 0.7
                base += max(0.0, (adx - adx_min)) * 0.2
                base += min(15.0, max(0.0, rvol - 1.0) * 8.0)
                return base
            return 0

        # --- Expanded library strategies (Option A wiring) ---

        if name == "TREND_RIDER":
            rsi_min = self._safe_float(conf.get("rsi_min", 52.0), default=52.0)
            adx_min = self._safe_float(conf.get("adx_min", 20.0), default=20.0)
            # Trend stack: price > EMA50 > EMA200
            if price > ema_50 > ema_200 and rsi >= rsi_min and adx >= adx_min:
                base = 65.0
                base += max(0.0, (rsi - rsi_min)) * 0.6
                base += max(0.0, (adx - adx_min)) * 0.4
                return base
            return 0

        if name == "PULLBACK_BUY":
            rsi_max = self._safe_float(conf.get("rsi_max", 45.0), default=45.0)
            adx_min = self._safe_float(conf.get("adx_min", 18.0), default=18.0)
            # In uptrend (EMA50>EMA200), pullback below EMA20 with RSI cooled
            if ema_50 > ema_200 and price < ema_20 and rsi <= rsi_max and adx >= adx_min:
                base = 60.0
                base += max(0.0, (rsi_max - rsi)) * 0.7
                base += max(0.0, (adx - adx_min)) * 0.3
                return base
            return 0

        if name == "DONCHIAN_BREAKOUT":
            rsi_min = self._safe_float(conf.get("rsi_min", 55.0), default=55.0)
            adx_min = self._safe_float(conf.get("adx_min", 20.0), default=20.0)
            if donchian_high > 0 and price > donchian_high and rsi >= rsi_min and adx >= adx_min:
                base = 66.0
                base += max(0.0, (rsi - rsi_min)) * 0.4
                base += max(0.0, (adx - adx_min)) * 0.4
                return base
            return 0

        if name == "ATR_BREAKOUT":
            adx_min = self._safe_float(conf.get("adx_min", 18.0), default=18.0)
            rsi_min = self._safe_float(conf.get("rsi_min", 50.0), default=50.0)
            lookback = int(self._safe_float(conf.get("lookback_len", 20), default=20))
            atr_mult = self._safe_float(conf.get("atr_mult", 0.8), default=0.8)

            prior_high = 0.0
            try:
                if isinstance(df_history, pd.DataFrame) and (not df_history.empty) and "high" in df_history.columns:
                    hh = pd.to_numeric(df_history["high"], errors="coerce")
                    if len(hh) >= lookback + 2:
                        prior_high = float(hh.tail(lookback + 1).iloc[:-1].max())
                    else:
                        prior_high = float(hh.max())
            except Exception:
                prior_high = 0.0

            trigger = prior_high + (atr * atr_mult) if (prior_high > 0 and atr > 0) else 0.0
            if trigger > 0 and price > trigger and rsi >= rsi_min and adx >= adx_min:
                base = 64.0
                base += max(0.0, (price - trigger) / max(1e-9, trigger)) * 200.0
                base += max(0.0, (adx - adx_min)) * 0.3
                return base
            return 0

        if name == "RSI2_REVERSAL":
            rsi2_buy = self._safe_float(conf.get("rsi2_buy", 5.0), default=5.0)
            if rsi2 > 0 and rsi2 <= rsi2_buy and price > ema_200:
                base = 58.0
                base += max(0.0, (rsi2_buy - rsi2)) * 2.0
                return base
            return 0

        if name == "Z_MEAN_REVERT":
            z_entry = self._safe_float(conf.get("z_entry", -2.0), default=-2.0)
            # Enter when zscore is sufficiently negative (oversold)
            if zscore != 0 and zscore <= z_entry and price > ema_200:
                base = 57.0
                base += max(0.0, abs(zscore - z_entry)) * 5.0
                return base
            return 0

        if name == "VWAP_REVERT":
            # If vwap_dev_pct is written as e.g. 0.8 meaning 0.8%, treat it as percent.
            dev_pct = self._safe_float(conf.get("vwap_dev_pct", 0.8), default=0.8)
            dev = dev_pct / 100.0 if dev_pct > 1 else dev_pct / 100.0
            # Always interpret as pct, not a 0-1 fraction.
            threshold = vwap * (1.0 - dev) if vwap > 0 else price
            rsi_buy = self._safe_float(conf.get("rsi_buy", 35.0), default=35.0)
            if price < threshold and rsi <= rsi_buy and price > ema_200:
                base = 56.0
                base += max(0.0, (threshold - price) / max(1e-9, threshold)) * 200.0
                return base
            return 0

        # Unknown strategy: do nothing
        return 0

    # --------------------
    # Risk / trade management parameters
    # --------------------
    def requires_confirmation(self, strategy_name):
        conf = self._get_conf(strategy_name)
        try:
            return str(conf.get('require_confirmation', 'False')).lower() == 'true'
        except Exception:
            return False

    def calculate_exit_prices(self, entry_price, atr, strategy_name):
        """Return (stop_price, take_profit_price) or (None, None) for crypto.

        Engine passes strat_name WITHOUT prefix.
        """
        conf = self._get_conf(strategy_name)
        if not conf:
            return None, None

        stop_loss_pct = self._safe_float(conf.get('stop_loss', 0.02), default=0.02)
        try:
            atr_stop_mult = float(self.config['CONFIGURATION'].get('backtest_atr_stop_mult', 2.0))
        except Exception:
            atr_stop_mult = 2.0
        try:
            atr_take_mult = float(self.config['CONFIGURATION'].get('backtest_atr_take_mult', 3.0))
        except Exception:
            atr_take_mult = 3.0

        if atr and atr > 0:
            stop_price = float(entry_price) - float(atr) * float(atr_stop_mult)
            take_price = float(entry_price) + float(atr) * float(atr_take_mult)
            return stop_price, take_price

        stop_price = float(entry_price) * (1 - stop_loss_pct)
        take_price = float(entry_price) * (1 + stop_loss_pct * 1.5)
        return stop_price, take_price


class WalletManager:
    # NOTE: Engine constructs WalletManager(config, db).
    # Keep this signature tolerant to avoid UI/engine import-time crashes.
    def __init__(self, config, db=None):
        self.config = config
        self.db = db
        self.daily_spent = 0

    def reset_daily_budget(self):
        self.daily_spent = 0

    def can_buy(self, symbol, price, current_equity):
        # Ensure we have a config section
        try:
            cfg = self.config['CONFIGURATION']
        except Exception:
            return True, "OK"

        max_daily = float(cfg.get('daily_budget', 1000))
        max_trades = int(float(cfg.get('max_open_trades', 5)))

        # If equity is below min, stop trading
        min_eq = float(cfg.get('min_equity', 1000))
        if current_equity < min_eq:
            return False, "Equity below minimum"

        if self.daily_spent + price > max_daily:
            return False, "Daily budget exceeded"

        # Trade count constraint is enforced by DB/engine; keep this lightweight
        return True, "OK"

    def get_trade_qty(self, price, atr, current_equity, ai_score=0.5):
        try:
            cfg = self.config['CONFIGURATION']
        except Exception:
            return 0

        base_risk_pct = float(cfg.get('risk_per_trade_pct', 1.0)) / 100.0
        max_qty = int(float(cfg.get('max_qty', 100)))

        # AI adjusts sizing modestly (0.5 neutral)
        ai_adj = max(0.5, min(1.5, float(ai_score) * 2.0))

        dollar_risk = current_equity * base_risk_pct * ai_adj

        # ATR-based sizing if ATR present, else simple notional
        if atr and atr > 0:
            stop_dist = max(atr * 2.0, price * 0.01)  # at least 1%
            qty = int(dollar_risk / stop_dist)
        else:
            qty = int(dollar_risk / max(price * 0.02, 0.01))

        qty = max(0, min(max_qty, qty))
        return qty
