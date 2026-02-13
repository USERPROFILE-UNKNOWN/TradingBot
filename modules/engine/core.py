"""Trading engine core.

This module contains the full TradingEngine implementation.
It was moved here during the v5.x refactor series to reduce blast radius.

"""

import alpaca_trade_api as tradeapi
import pandas_ta as ta
import requests
import os
import json
import uuid
import hashlib
import math
import time
import pandas as pd
import threading
from datetime import datetime, timedelta, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..strategies import StrategyOptimizer, WalletManager
from ..sentiment import NewsSentinel 
from ..ai import AI_Oracle 
from ..utils import APP_VERSION, APP_RELEASE, get_paths

class TradingEngine:
    def __init__(self, config, db, log_callback, update_chart_callback):
        self.config = config
        self.db = db
        self.log = log_callback
        self.update_chart = update_chart_callback 
        self.agent_master = None
        self.active = False
        self.api = None
        self.optimizer = StrategyOptimizer(config, db) 
        self.wallet = WalletManager(config, db)
        self._log_throttle = {}
        # Release C2: periodic snapshot logging
        self._last_snapshot_ts = 0.0
        self._scan_cycle = 0
        self._last_scan_stats = None
        self._min_log_level = 20
        self._log_snapshot_interval_sec = 300
        self._log_candidate_lines = False
        self._log_decision_rejects = False
        self._log_scan_summary = True
        self._log_guardian_banner = False
        self._reload_logging_config()

        # Release D1: order lifecycle tracking (prevents DB “ghost trades”)
        self.pending_orders = {}  # order_id -> meta
        self._pending_symbols = set()
        self._live_entry_ttl_sec = 120
        self._live_order_poll_sec = 5
        self._live_cancel_unfilled_entries = True
        self._log_order_lifecycle = True
        self._reload_execution_config()
        self._reload_e3_config()
        self._reload_e4_config()

        # Release E3: run summary + data freshness
        self._paths = get_paths()
        self._run_summary_enabled = True
        self._run_summary_prefix = 'SUMMARY'
        self._stale_bar_seconds_threshold = 180
        self._stale_bar_log_each_symbol = False
        self._last_bar_epoch = {}  # SYMBOL -> epoch seconds
        self._session_started_at = None
        self._session_id = None
        self._stop_reason = None
        self._order_stats = {
            'submitted': 0,
            'filled': 0,
            'canceled': 0,
            'rejected': 0,
            'expired': 0,
            'ttl_canceled': 0,
        }
        self._reload_e3_config()

        # Release E5: Promotion and safeguards (watchdog / kill-switch hardening)
        self._entries_disabled = False
        self._entries_disabled_reason = None
        self._e5_api_error_events = deque()
        self._e5_reject_events = deque()
        self._e5_ttl_cancel_events = deque()
        self._e5_last_watchdog_ts = 0.0
        self._reload_e5_config()

        self.start_equity = 0.0
        self.current_equity = 0.0 
        
        try:
            base_amt = float(config['CONFIGURATION'].get('amount_to_trade', 2000))
            base_loss = float(config['CONFIGURATION'].get('max_daily_loss', 100))
            self.kill_switch_ratio = base_loss / base_amt 
        except:
            self.kill_switch_ratio = 0.05

        self.pending_confirmations = {}
        self.market_status = "UNKNOWN" 
        self.last_regime_check = datetime.min
        self.last_telegram_update_id = 0 
        self.connect_api()
        
        if self.api:
            self.sentinel = NewsSentinel(self.api)
            
        # v3.9.16 FIX: Pass log_callback so AI speaks to UI
        self.ai = AI_Oracle(self.db, self.config, self.log)
        threading.Thread(target=self.ai_training_thread, daemon=True).start()

    def set_agent_master(self, agent_master):
        self.agent_master = agent_master

    def _publish_agent_event(self, event_type: str, payload: dict | None = None):
        try:
            if self.agent_master:
                self.agent_master.publish(event_type, payload or {})
        except Exception:
            pass

    def ai_training_thread(self):
        # v3.9.16: Using the AI's internal logger now
        self.ai.train_model()

    # --------------------
    # Release C: Structured logging helper (backward-compatible)
    # --------------------
    def _cfg(self, key, default=None):
        try:
            if 'CONFIGURATION' in self.config:
                return self.config['CONFIGURATION'].get(key, default)
        except Exception:
            pass
        return default

    # --- Data feed selection (IEX vs SIP) ---
    def _equity_feed(self):
        """Return the configured Alpaca equity data feed (defaults to IEX).

        Free accounts generally only support IEX. SIP requires a paid subscription.
        """
        try:
            feed = str(self._cfg('equity_data_feed', 'iex') or 'iex').strip().lower()
        except Exception:
            feed = 'iex'
        return feed if feed in ('iex', 'sip') else 'iex'

    def _api_get_bars_equity(self, symbol, timeframe, **kwargs):
        """Wrapper around alpaca_trade_api.get_bars that forces feed= for equities when supported.

        Some alpaca_trade_api versions do not accept the feed= kwarg; this wrapper auto-falls back.
        """
        feed = self._equity_feed()
        try:
            return self.api.get_bars(symbol, timeframe, feed=feed, **kwargs)
        except TypeError:
            return self.api.get_bars(symbol, timeframe, **kwargs)

    def _cfg_bool(self, key, default=False):
        v = self._cfg(key, default)
        try:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        except Exception:
            return bool(default)

    def _cfg_int(self, key, default=0):
        v = self._cfg(key, default)
        try:
            return int(float(v))
        except Exception:
            return int(default)

    def _cfg_float(self, key, default=0.0):
        v = self._cfg(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def _parse_level(self, level_str, default=20):
        try:
            s = str(level_str or '').strip().upper()
        except Exception:
            return default
        mapping = {'DEBUG': 10, 'INFO': 20, 'WARN': 30, 'WARNING': 30, 'ERROR': 40}
        return mapping.get(s, default)

    def _reload_logging_config(self):
        """Reload log verbosity knobs from config.ini. Safe if keys are absent."""
        try:
            self._min_log_level = self._parse_level(self._cfg('log_level', 'INFO'), default=20)
            self._log_snapshot_interval_sec = max(0, self._cfg_int('log_snapshot_interval_sec', 300))
            self._log_candidate_lines = self._cfg_bool('log_candidate_lines', False)
            self._log_decision_rejects = self._cfg_bool('log_decision_rejects', False)
            self._log_scan_summary = self._cfg_bool('log_scan_summary', True)
            self._log_guardian_banner = self._cfg_bool('log_guardian_banner', False)
        except Exception:
            # Keep safe defaults
            self._min_log_level = 20
            self._log_snapshot_interval_sec = 300
            self._log_candidate_lines = False
            self._log_decision_rejects = False
            self._log_scan_summary = True
            self._log_guardian_banner = False

    def _reload_execution_config(self):
        """Reload execution / order lifecycle knobs from config.ini. Safe if keys are absent."""
        try:
            self._live_entry_ttl_sec = max(0, self._cfg_int('live_entry_ttl_seconds', 120))
            self._live_order_poll_sec = max(1, self._cfg_int('live_order_poll_interval_seconds', 5))
            self._live_cancel_unfilled_entries = self._cfg_bool('live_cancel_unfilled_entries', True)
            self._log_order_lifecycle = self._cfg_bool('log_order_lifecycle', True)
            # v5.12.5: reconciliation safeguards
            self._reconcile_interval_sec = max(10, self._cfg_int('reconcile_interval_seconds', 60))
            self._reconcile_confirmations = max(1, self._cfg_int('reconcile_confirmations', 2))
            self._reconcile_halt_on_mismatch = self._cfg_bool('reconcile_halt_on_mismatch', True)
        except Exception:
            self._live_entry_ttl_sec = 120
            self._live_order_poll_sec = 5
            self._live_cancel_unfilled_entries = True
            self._log_order_lifecycle = True
            self._reconcile_interval_sec = 60
            self._reconcile_confirmations = 2
            self._reconcile_halt_on_mismatch = True
    def _reload_e3_config(self):
        """Reload Release E3 knobs from config.ini. Safe if keys are absent."""
        try:
            self._run_summary_enabled = self._cfg_bool('run_summary_enabled', True)
            try:
                prefix = str(self._cfg('run_summary_prefix', 'SUMMARY') or 'SUMMARY').strip()
            except Exception:
                prefix = 'SUMMARY'
            self._run_summary_prefix = prefix or 'SUMMARY'

            self._stale_bar_seconds_threshold = max(0, self._cfg_int('stale_bar_seconds_threshold', 180))
            self._stale_bar_log_each_symbol = self._cfg_bool('stale_bar_log_each_symbol', False)
        except Exception:
            self._run_summary_enabled = True
            self._run_summary_prefix = 'SUMMARY'
            self._stale_bar_seconds_threshold = 180
            self._stale_bar_log_each_symbol = False



    # --------------------
    # Release E4: Diversification framework (regime + playbooks)
    # --------------------
    def _reload_e4_config(self):
        """Reload Release E4 knobs from config.ini. Safe if keys are absent."""
        try:
            self._e4_enabled = self._cfg_bool('e4_enabled', False)
            self._e4_mtf_enabled = self._cfg_bool('e4_mtf_enabled', False)

            # Multi-timeframe confirmation settings
            try:
                raw_tfs = str(self._cfg('e4_mtf_timeframes', '5Min,15Min') or '5Min,15Min')
                tfs = [t.strip() for t in raw_tfs.split(',') if t.strip()]
            except Exception:
                tfs = ['5Min', '15Min']
            allowed = {'5Min', '15Min'}
            self._e4_mtf_timeframes = [t for t in tfs if t in allowed] or ['5Min', '15Min']

            self._e4_mtf_refresh_seconds = max(5, self._cfg_int('e4_mtf_refresh_seconds', 120))
            self._e4_mtf_ema_fast = max(5, self._cfg_int('e4_mtf_ema_fast', 50))
            self._e4_mtf_ema_slow = max(self._e4_mtf_ema_fast + 1, self._cfg_int('e4_mtf_ema_slow', 200))
            self._e4_mtf_rsi_min_momentum = float(self._cfg('e4_mtf_rsi_min_momentum', 50) or 50)

            # Correlation clusters
            self._e4_max_positions_per_cluster = max(0, self._cfg_int('e4_max_positions_per_cluster', 0))
            raw_clusters = str(self._cfg('e4_correlation_clusters', '') or '')
            self._e4_clusters = self._e4_parse_clusters(raw_clusters)

            # Cache for multi-timeframe bars to limit API calls
            if not hasattr(self, '_e4_mtf_cache'):
                self._e4_mtf_cache = {}  # (symbol, timeframe, is_crypto) -> (fetched_epoch, df)

        except Exception:
            # Fail-open: disable E4 if anything goes wrong parsing config
            self._e4_enabled = False
            self._e4_mtf_enabled = False
            self._e4_mtf_timeframes = ['5Min', '15Min']
            self._e4_mtf_refresh_seconds = 120
            self._e4_mtf_ema_fast = 50
            self._e4_mtf_ema_slow = 200
            self._e4_mtf_rsi_min_momentum = 50
            self._e4_max_positions_per_cluster = 0
            self._e4_clusters = {}
            if not hasattr(self, '_e4_mtf_cache'):
                self._e4_mtf_cache = {}


    # --------------------
    # Release E5: Promotion and safeguards (watchdog / entry halts)
    # --------------------
    def _reload_e5_config(self):
        """Reload Release E5 safeguard knobs from config.ini (safe if keys are absent)."""
        try:
            # Enable/disable & enforcement scope
            self._e5_enabled = self._cfg_bool('e5_enabled', True)
            self._e5_enforce_on_paper = self._cfg_bool('e5_enforce_on_paper', False)
            self._e5_enforce_on_live = self._cfg_bool('e5_enforce_on_live', True)

            # Watchdog cadence and event window
            self._e5_watchdog_interval_sec = max(5, self._cfg_int('e5_watchdog_interval_sec', 30))
            self._e5_event_window_sec = max(60, self._cfg_int('e5_event_window_sec', 900))

            # Thresholds (0 disables that trigger)
            self._e5_max_rejects_in_window = max(0, self._cfg_int('e5_max_rejects_in_window', 10))
            self._e5_max_api_errors_in_window = max(0, self._cfg_int('e5_max_api_errors_in_window', 20))
            self._e5_max_ttl_cancels_in_window = max(0, self._cfg_int('e5_max_ttl_cancels_in_window', 20))

            # Halt actions
            self._e5_halt_cancel_pending_orders = self._cfg_bool('e5_halt_cancel_pending_orders', True)
            self._e5_halt_liquidate = self._cfg_bool('e5_halt_liquidate', False)
            self._e5_halt_stop_engine = self._cfg_bool('e5_halt_stop_engine', False)

            # Ensure event queues exist (timestamps as epoch seconds)
            if not hasattr(self, '_e5_api_error_events') or self._e5_api_error_events is None:
                self._e5_api_error_events = deque()
            if not hasattr(self, '_e5_reject_events') or self._e5_reject_events is None:
                self._e5_reject_events = deque()
            if not hasattr(self, '_e5_ttl_cancel_events') or self._e5_ttl_cancel_events is None:
                self._e5_ttl_cancel_events = deque()

        except Exception:
            # Fail-open (no halts) if parsing fails
            self._e5_enabled = False
            self._e5_enforce_on_paper = False
            self._e5_enforce_on_live = False
            self._e5_watchdog_interval_sec = 30
            self._e5_event_window_sec = 900
            self._e5_max_rejects_in_window = 0
            self._e5_max_api_errors_in_window = 0
            self._e5_max_ttl_cancels_in_window = 0
            self._e5_halt_cancel_pending_orders = False
            self._e5_halt_liquidate = False
            self._e5_halt_stop_engine = False
            if not hasattr(self, '_e5_api_error_events') or self._e5_api_error_events is None:
                self._e5_api_error_events = deque()
            if not hasattr(self, '_e5_reject_events') or self._e5_reject_events is None:
                self._e5_reject_events = deque()
            if not hasattr(self, '_e5_ttl_cancel_events') or self._e5_ttl_cancel_events is None:
                self._e5_ttl_cancel_events = deque()

    def _e5_is_enforced(self):
        """Return True when safeguards should be enforced for the current session (paper/live)."""
        try:
            if not bool(getattr(self, '_e5_enabled', False)):
                return False

            base_url = ''
            try:
                base_url = str(self.config['KEYS'].get('base_url', '')).lower()
            except Exception:
                base_url = ''

            is_paper = ('paper' in base_url) or ('paper-api' in base_url)

            if is_paper:
                return bool(getattr(self, '_e5_enforce_on_paper', False))
            return bool(getattr(self, '_e5_enforce_on_live', True))
        except Exception:
            return False

    def _e5__prune_events(self, dq, now_ts):
        """Prune an event deque to the configured E5 window."""
        try:
            window = float(getattr(self, '_e5_event_window_sec', 0) or 0)
            if window <= 0:
                return
            cutoff = float(now_ts) - window
            while dq and float(dq[0]) < cutoff:
                dq.popleft()
        except Exception:
            pass

    def _e5_note_api_error(self, _kind=None):
        """Record an API error occurrence (best-effort)."""
        try:
            now_ts = time.time()
            if not hasattr(self, '_e5_api_error_events') or self._e5_api_error_events is None:
                self._e5_api_error_events = deque()
            self._e5_api_error_events.append(float(now_ts))
            self._e5__prune_events(self._e5_api_error_events, now_ts)
        except Exception:
            pass

    def _e5_note_reject(self, _symbol=None):
        """Record an order rejection occurrence (best-effort)."""
        try:
            now_ts = time.time()
            if not hasattr(self, '_e5_reject_events') or self._e5_reject_events is None:
                self._e5_reject_events = deque()
            self._e5_reject_events.append(float(now_ts))
            self._e5__prune_events(self._e5_reject_events, now_ts)
        except Exception:
            pass

    def _e5_note_ttl_cancel(self, _symbol=None):
        """Record a TTL cancel occurrence (best-effort)."""
        try:
            now_ts = time.time()
            if not hasattr(self, '_e5_ttl_cancel_events') or self._e5_ttl_cancel_events is None:
                self._e5_ttl_cancel_events = deque()
            self._e5_ttl_cancel_events.append(float(now_ts))
            self._e5__prune_events(self._e5_ttl_cancel_events, now_ts)
        except Exception:
            pass

    def _e5_watchdog_tick(self):
        """Periodic safeguard watchdog. Can halt new entries and optionally take cleanup actions."""
        try:
            if not self._e5_is_enforced():
                return

            now_ts = time.time()
            interval = float(getattr(self, '_e5_watchdog_interval_sec', 30) or 30)
            last_ts = float(getattr(self, '_e5_last_watchdog_ts', 0.0) or 0.0)
            if (now_ts - last_ts) < interval:
                return
            self._e5_last_watchdog_ts = now_ts

            # Ensure queues exist and prune them
            if not hasattr(self, '_e5_api_error_events') or self._e5_api_error_events is None:
                self._e5_api_error_events = deque()
            if not hasattr(self, '_e5_reject_events') or self._e5_reject_events is None:
                self._e5_reject_events = deque()
            if not hasattr(self, '_e5_ttl_cancel_events') or self._e5_ttl_cancel_events is None:
                self._e5_ttl_cancel_events = deque()

            self._e5__prune_events(self._e5_api_error_events, now_ts)
            self._e5__prune_events(self._e5_reject_events, now_ts)
            self._e5__prune_events(self._e5_ttl_cancel_events, now_ts)

            api_err = len(self._e5_api_error_events)
            rej = len(self._e5_reject_events)
            ttl = len(self._e5_ttl_cancel_events)

            # Determine trigger(s)
            triggers = []
            if int(getattr(self, '_e5_max_api_errors_in_window', 0) or 0) > 0 and api_err >= int(self._e5_max_api_errors_in_window):
                triggers.append(f"API_ERRORS {api_err}/{int(self._e5_max_api_errors_in_window)}")
            if int(getattr(self, '_e5_max_rejects_in_window', 0) or 0) > 0 and rej >= int(self._e5_max_rejects_in_window):
                triggers.append(f"REJECTS {rej}/{int(self._e5_max_rejects_in_window)}")
            if int(getattr(self, '_e5_max_ttl_cancels_in_window', 0) or 0) > 0 and ttl >= int(self._e5_max_ttl_cancels_in_window):
                triggers.append(f"TTL_CANCELS {ttl}/{int(self._e5_max_ttl_cancels_in_window)}")

            if not triggers:
                return

            # If already halted, don't spam
            if getattr(self, '_entries_disabled', False):
                return

            reason = "E5_WATCHDOG: " + ", ".join(triggers)
            self._entries_disabled = True
            self._entries_disabled_reason = reason

            try:
                self._emit(
                    f"⛔ Safeguard halt: new entries disabled | {reason}",
                    level="ERROR",
                    category="RISK",
                    throttle_key="e5_halt",
                    throttle_sec=60
                )
            except Exception:
                pass
            try:
                self.send_telegram(f"⛔ Safeguard halt: new entries disabled | {reason}")
            except Exception:
                pass

            # Optional cleanup actions
            if bool(getattr(self, '_e5_halt_cancel_pending_orders', False)):
                try:
                    self.cancel_all_pending_orders()
                except Exception:
                    pass
            if bool(getattr(self, '_e5_halt_liquidate', False)):
                try:
                    self.liquidate_all()
                except Exception:
                    pass
            if bool(getattr(self, '_e5_halt_stop_engine', False)):
                try:
                    self.stop(reason="E5_WATCHDOG")
                except Exception:
                    pass

        except Exception:
            # Fail-open: never crash the engine because of the watchdog
            return


    def _e4_parse_clusters(self, raw: str):
        """Parse cluster config string into {cluster_name: set(symbols)}."""
        clusters = {}
        try:
            if not raw:
                return clusters
            groups = [g.strip() for g in str(raw).split('|') if g.strip()]
            for group in groups:
                if ':' not in group:
                    continue
                name, syms = group.split(':', 1)
                name = name.strip().upper()
                sym_list = [s.strip().upper() for s in syms.split(',') if s.strip()]
                if not name or not sym_list:
                    continue
                clusters[name] = set(sym_list)
        except Exception:
            return {}
        return clusters

    def _e4_strategy_playbook(self, strat_name: str) -> str:
        """Map a strategy name to a coarse playbook label."""
        try:
            s = (strat_name or '').lower()
        except Exception:
            return 'GENERIC'
        # heuristic mapping; safe default is GENERIC (minimal gating)
        if any(k in s for k in ['breakout', 'momentum', 'trend', 'surge']):
            return 'MOMENTUM'
        if any(k in s for k in ['mean', 'reversion', 'dip', 'bounce', 'pullback']):
            return 'MEAN_REVERSION'
        return 'GENERIC'

    def _e4_count_cluster_exposure(self, cluster_name: str) -> int:
        """Count distinct open/pending symbols in a cluster."""
        try:
            cluster = self._e4_clusters.get(str(cluster_name).upper(), set())
            if not cluster:
                return 0
        except Exception:
            return 0

        active_syms = set()
        pending_syms = set()
        try:
            trades = self.db.get_active_trades() or {}
            for sym in (trades.keys() if isinstance(trades, dict) else []):
                active_syms.add(str(sym).upper())
        except Exception:
            pass
        try:
            for sym in getattr(self, '_pending_symbols', set()) or set():
                pending_syms.add(str(sym).upper())
        except Exception:
            pass

        exposed = (active_syms | pending_syms) & set(cluster)
        return len(exposed)

    def _e4_cluster_allows_entry(self, symbol: str):
        """Return (ok, reason). Enforces max positions per cluster if configured."""
        try:
            if not getattr(self, '_e4_enabled', False):
                return True, 'E4 disabled'
            cap = int(getattr(self, '_e4_max_positions_per_cluster', 0) or 0)
            if cap <= 0:
                return True, 'Cluster cap disabled'
            sym = str(symbol).upper()
            # find clusters containing this symbol
            hit = [name for name, members in (self._e4_clusters or {}).items() if sym in members]
            if not hit:
                return True, 'No cluster'
            for cluster_name in hit:
                exposure = self._e4_count_cluster_exposure(cluster_name)
                if exposure >= cap:
                    return False, f'Cluster cap: {cluster_name} ({exposure}/{cap})'
            return True, 'Cluster OK'
        except Exception:
            return True, 'Cluster check bypass (error)'

    def _e4_get_mtf_bars(self, symbol: str, timeframe: str, is_crypto: bool):
        """Fetch bars for a higher timeframe with caching. Returns a DataFrame or None."""
        try:
            if not getattr(self, '_e4_enabled', False) or not getattr(self, '_e4_mtf_enabled', False):
                return None

            key = (str(symbol).upper(), str(timeframe), bool(is_crypto))
            now = time.time()
            ttl = float(getattr(self, '_e4_mtf_refresh_seconds', 120) or 120)
            try:
                cached = self._e4_mtf_cache.get(key)
            except Exception:
                cached = None
            if cached:
                fetched_epoch, df = cached
                if (now - float(fetched_epoch)) <= ttl:
                    return df

            # Fetch fresh
            if is_crypto:
                df = self.retry_api_call(self.api.get_crypto_bars, symbol, timeframe, limit=300).df
            else:
                df = self.retry_api_call(self._api_get_bars_equity, symbol, timeframe, limit=300).df

            if df is None or df.empty:
                return None
            df = df[~df.index.duplicated(keep='first')]

            try:
                self._e4_mtf_cache[key] = (now, df)
            except Exception:
                pass
            return df
        except Exception:
            return None

    def _e4_get_mtf_context(self, symbol: str, is_crypto: bool):
        """Compute minimal MTF context needed for gating."""
        ctx = {}
        try:
            fast = int(getattr(self, '_e4_mtf_ema_fast', 50) or 50)
            slow = int(getattr(self, '_e4_mtf_ema_slow', 200) or 200)

            # 15Min trend filter
            df15 = self._e4_get_mtf_bars(symbol, '15Min', is_crypto)
            if df15 is not None and not df15.empty and len(df15) >= (slow + 10):
                ema_fast_15 = ta.ema(df15['close'], length=fast)
                ema_slow_15 = ta.ema(df15['close'], length=slow)
                if ema_fast_15 is not None and ema_slow_15 is not None:
                    ctx['trend_up_15'] = float(ema_fast_15.iloc[-1]) >= float(ema_slow_15.iloc[-1])
            # 5Min RSI for momentum
            df5 = self._e4_get_mtf_bars(symbol, '5Min', is_crypto)
            if df5 is not None and not df5.empty and len(df5) >= 30:
                rsi5 = ta.rsi(df5['close'], length=14)
                if rsi5 is not None:
                    ctx['rsi_5'] = float(rsi5.iloc[-1])
        except Exception:
            return ctx
        return ctx

    def _e4_mtf_allows_trade(self, symbol: str, strat_name: str, is_crypto: bool):
        """Return (ok, reason). Uses 15Min trend filter and optional 5Min RSI momentum gate."""
        try:
            if not getattr(self, '_e4_enabled', False):
                return True, 'E4 disabled'
            if not getattr(self, '_e4_mtf_enabled', False):
                return True, 'MTF disabled'

            playbook = self._e4_strategy_playbook(strat_name)
            ctx = self._e4_get_mtf_context(symbol, is_crypto)

            # If context is incomplete (API hiccup), fail-open to preserve baseline behavior.
            if 'trend_up_15' not in ctx:
                return True, 'MTF bypass (no 15Min trend)'
            if not ctx['trend_up_15']:
                return False, 'MTF reject: 15Min trend down'

            if playbook == 'MOMENTUM':
                min_rsi = float(getattr(self, '_e4_mtf_rsi_min_momentum', 50) or 50)
                rsi5 = ctx.get('rsi_5', None)
                if rsi5 is None:
                    return True, 'MTF bypass (no 5Min RSI)'
                if float(rsi5) < float(min_rsi):
                    return False, f'MTF reject: 5Min RSI {rsi5:.1f} < {min_rsi:.1f}'

            return True, 'MTF OK'
        except Exception:
            return True, 'MTF bypass (error)'


    def _redact(self, text):
        """Best-effort secret redaction for any log content."""
        try:
            s = str(text)
        except Exception:
            return "***REDACTED***"

        try:
            keys_sec = None
            if isinstance(self.config, dict) and 'KEYS' in self.config:
                keys_sec = self.config['KEYS']
            if keys_sec is not None:
                for k in ('alpaca_key', 'alpaca_secret', 'telegram_token', 'telegram_chat_id'):
                    try:
                        v = str(keys_sec.get(k, '')).strip()
                        if v and v in s:
                            s = s.replace(v, '***REDACTED***')
                    except Exception:
                        pass
        except Exception:
            pass

        return s

    def _emit(self, msg, level="INFO", category=None, symbol=None, throttle_key=None, throttle_sec=0):
        """Emit a log record to UI. Works with both legacy and structured log callbacks."""
        # Honor configured minimum log level
        try:
            lvl_num = self._parse_level(level, default=20)
            if lvl_num < int(self._min_log_level):
                return
        except Exception:
            pass

        try:
            if throttle_key and throttle_sec:
                now = time.time()
                last = self._log_throttle.get(throttle_key, 0)
                if (now - last) < float(throttle_sec):
                    return
                self._log_throttle[throttle_key] = now
        except Exception:
            pass

        # Agent event mapping for observability/governance.
        try:
            cat = str(category or '').upper()
            txt = str(msg)
            if cat == 'ORDER' and 'REJECT' in txt.upper():
                self._publish_agent_event('ORDER_REJECTED', {'symbol': symbol, 'message': txt})
            if cat == 'RISK':
                self._publish_agent_event('RISK_BREACH', {'symbol': symbol, 'message': txt})
            if cat == 'DATA':
                self._publish_agent_event('DATA_GAP', {'symbol': symbol, 'message': txt})
        except Exception:
            pass

        # If the UI supports structured logging, use it; otherwise prefix the string.
        try:
            self.log(msg, level=level, category=category, symbol=symbol)
            return
        except TypeError:
            pass
        except Exception:
            pass

        try:
            parts = []
            try:
                lvl = (level or "INFO").upper()
                if lvl and lvl != "INFO":
                    parts.append(lvl)
            except Exception:
                pass
            if category:
                parts.append(str(category).upper())
            if symbol:
                parts.append(str(symbol).upper())

            prefix = " ".join([f"[{p}]" for p in parts])
            final = f"{prefix} {msg}".strip() if prefix else str(msg)
            self.log(final)
        except Exception:
            try:
                self.log(str(msg))
            except Exception:
                pass



    # v5.12.6 updateA: structured execution packet logger (replay harness)
    def _log_exec_packet(self, *, symbol, side, phase, decision_id=None, qty=None, price=None, order_id=None, client_order_id=None, broker_status=None, payload=None):
        try:
            if not hasattr(self.db, 'log_execution_packet'):
                return
            self.db.log_execution_packet(
                symbol=symbol,
                side=side,
                phase=phase,
                decision_id=decision_id,
                qty=qty,
                price=price,
                order_id=order_id,
                client_order_id=client_order_id,
                broker_status=broker_status,
                payload=payload or {},
            )
        except Exception:
            pass
    def _emit_snapshot_if_due(self, force=False):
        """Periodic engine snapshot to improve Live Log observability without UI changes."""
        if not self._log_snapshot_interval_sec:
            return
        now = time.time()
        if (not force) and self._last_snapshot_ts and (now - self._last_snapshot_ts) < float(self._log_snapshot_interval_sec):
            return
        self._last_snapshot_ts = now

        try:
            trades = self.db.get_active_trades() or {}
        except Exception:
            trades = {}
        try:
            pending = len(self.pending_confirmations)
        except Exception:
            pending = 0
        try:
            pl = float(self.current_equity) - float(self.start_equity)
        except Exception:
            pl = 0.0

        # Release E3: data freshness snapshot (last bar timestamps)
        stale_count = 0
        stale_total = 0
        oldest_age = None
        try:
            stale_total = len(getattr(self, '_last_bar_epoch', {}) or {})
            threshold = float(getattr(self, '_stale_bar_seconds_threshold', 0) or 0)
            if threshold > 0 and stale_total > 0:
                now_epoch = float(now)
                ages = []
                for sym, epoch in (getattr(self, '_last_bar_epoch', {}) or {}).items():
                    try:
                        age = max(0.0, now_epoch - float(epoch))
                        ages.append((str(sym).upper(), age))
                    except Exception:
                        pass
                if ages:
                    oldest_age = max(a for _, a in ages)
                    stale_count = sum(1 for _, a in ages if a > threshold)
                    if getattr(self, '_stale_bar_log_each_symbol', False) and stale_count:
                        stale_syms = [s for s, a in ages if a > threshold]
                        msg = '⚠️ Stale bars detected: ' + ', '.join(stale_syms[:10])
                        self._emit(msg, level='WARN', category='DATA')
        except Exception:
            pass

        parts = [
            f"Regime={self.market_status}",
            f"Equity=${self.current_equity:.2f}",
            f"P/L=${pl:.2f}",
            f"Active={len(trades)}",
            f"PendingConf={pending}",
            f"PendingOrders={len(getattr(self, 'pending_orders', {}) or {})}",
            f"PendingSyms={len(getattr(self, '_pending_symbols', set()) or set())}",
            f"Cycle={int(getattr(self, '_scan_cycle', 0) or 0)}",
        ]
        if stale_total > 0:
            parts.append(f"StaleBars={stale_count}/{stale_total}")
            if oldest_age is not None:
                parts.append(f"OldestBarAge={int(oldest_age)}s")

        if isinstance(self._last_scan_stats, dict):
            try:
                parts.append(f"Scan: sym={self._last_scan_stats.get('symbols', 0)} opp={self._last_scan_stats.get('opps', 0)} buys={self._last_scan_stats.get('buys', 0)}")
            except Exception:
                pass

        self._emit("📊 Snapshot | " + " | ".join(parts), level="INFO", category="SNAPSHOT")


    def _record_last_bar_epoch(self, symbol, ts):
        """Record last bar timestamp (epoch seconds) for health checks."""
        try:
            sym = str(symbol).upper()
        except Exception:
            sym = symbol
        try:
            if ts is None:
                return
            if isinstance(ts, datetime):
                dt = ts
            else:
                dt = pd.to_datetime(ts, utc=True).to_pydatetime()
            self._last_bar_epoch[sym] = float(dt.timestamp())
        except Exception:
            pass
    def retry_api_call(self, func, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionResetError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                # Release E5: note API instability (best-effort, fail-open)
                try:
                    self._e5_note_api_error(type(e).__name__)
                except Exception:
                    pass
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
            except Exception as e:
                # Release E5: note unexpected API errors (best-effort, fail-open)
                try:
                    self._e5_note_api_error(type(e).__name__)
                except Exception:
                    pass
                raise e


    def connect_api(self):
        try:
            self.api = tradeapi.REST(
                key_id=self.config['KEYS']['alpaca_key'].strip(),
                secret_key=self.config['KEYS']['alpaca_secret'].strip(),
                base_url=self.config['KEYS'].get('base_url', 'https://paper-api.alpaca.markets').strip(),
                api_version='v2'
            )
            acct = self.retry_api_call(self.api.get_account)
            self.start_equity = float(acct.last_equity) 
            self.current_equity = float(acct.equity)
            self._emit("✅ API Connected", category="SYSTEM")
            self.sync_positions() 
        except Exception as e:
            self._emit(f"❌ API Error: {self._redact(e)}", level="ERROR", category="SYSTEM")


    def check_market_regime(self):
        # Broad market regime using a benchmark (default SPY).
        # Sets self.market_status to: BULL, BEAR, CHOP, HIGH_VOL (or UNKNOWN).
        try:
            # throttle (default: 30 minutes)
            now_s = time.time()
            if now_s - getattr(self, '_last_regime_check', 0) < 1800:
                return
            self._last_regime_check = now_s

            bench = str(self._cfg('market_regime_symbol', 'SPY') or 'SPY').strip().upper()

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=500)

            bars = self._api_get_bars_equity(
                bench,
                "1Day",
                start=start_dt.isoformat().replace("+00:00", "Z"),
                end=end_dt.isoformat().replace("+00:00", "Z"),
                limit=600,
                adjustment='raw'
            )

            df = getattr(bars, 'df', None)
            if df is None or df.empty:
                self.market_status = 'UNKNOWN'
                return

            df = df.reset_index()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.sort_values(by='timestamp')

            # Indicators
            df['sma_200'] = df['close'].rolling(200).mean()
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            try:
                df['adx_14'] = adx_df.iloc[:, 0]
            except Exception:
                df['adx_14'] = 0.0

            last = df.iloc[-1]
            price = float(last['close']) if pd.notna(last.get('close', None)) else 0.0
            sma200 = float(last['sma_200']) if pd.notna(last.get('sma_200', None)) else price
            adx14 = float(last.get('adx_14', 0.0) or 0.0)
            atr14 = float(last.get('atr_14', 0.0) or 0.0)
            atr_pct = (atr14 / price) if price > 0 else 0.0

            # Thresholds (configurable)
            chop_adx_max = self._cfg_float('regime_chop_adx_max', 18.0)
            highvol_atr_pct_min = self._cfg_float('regime_highvol_atr_pct_min', 0.03)

            trend = 'BULL' if price >= sma200 else 'BEAR'
            regime = trend
            if adx14 > 0 and adx14 < chop_adx_max:
                regime = 'CHOP'
            elif atr_pct > 0 and atr_pct >= highvol_atr_pct_min:
                regime = 'HIGH_VOL'

            self.market_status = regime
            self._emit(
                f"[REGIME] {bench} | price={price:.2f} | sma200={sma200:.2f} | adx14={adx14:.2f} | atr%={atr_pct*100:.2f}% => {self.market_status}",
                category='SYSTEM'
            )

        except Exception as e:
            self.market_status = getattr(self, 'market_status', 'UNKNOWN') or 'UNKNOWN'
            try:
                self._emit(f"[REGIME] ⚠️ regime check failed: {e}", category='SYSTEM')
            except Exception:
                pass

    def sync_positions(self):
        try:
            self._emit("🔄 Syncing Alpaca positions to Database...", category="SYNC")
            alpaca_positions = self.retry_api_call(self.api.list_positions)
            db_trades = self.db.get_active_trades()
            
            for p in alpaca_positions:
                symbol = p.symbol
                qty = float(p.qty) 
                entry_price = float(p.avg_entry_price)
                if symbol not in db_trades:
                    self.db.log_trade_entry(symbol, qty, entry_price, "IMPORTED")
                    self._emit(f"📥 Imported missing trade: {symbol} ({qty} units)", level="INFO", category="SYNC", symbol=symbol, throttle_key=f"imp_{symbol}", throttle_sec=60)
            
            active_symbols = [p.symbol for p in alpaca_positions]
            for symbol in db_trades:
                if symbol not in active_symbols:
                    # Remove DB-only "ghost" rows without polluting trade_history with exit_price=0
                    try:
                        self.db.remove_active_trade(symbol)
                    except Exception:
                        # Fallback for older DB layer
                        try:
                            self.db.close_trade(symbol, 0)
                        except Exception:
                            pass
                    self._emit(f"👻 Removed ghost trade: {symbol}", level="WARN", category="SYNC", symbol=symbol, throttle_key=f"ghost_{symbol}", throttle_sec=60)
            self._emit("✅ Sync Complete.", category="SYNC")
            # Release D1: also attempt to reconstruct any pending orders after restart
            self.sync_open_orders()
        except Exception as e:
            self._emit(f"⚠️ Sync Failed: {self._redact(e)}", level="WARN", category="SYNC", throttle_key="sync_fail", throttle_sec=60)

    # --------------------
    # Release D1: Pending order lifecycle (prevents DB trades before fill)
    # --------------------
    def _make_client_order_id(self, symbol, strat_name, side='B', idem_suffix=None):
        """Create a short client_order_id so open orders can be reconstructed after restart.

        v5.12.5: idempotency
          - Use a deterministic suffix (idem_suffix) when available so retries / re-entrant scans
            do not accidentally create duplicate orders on API hiccups.
        """
        try:
            sym = str(symbol).upper().replace('/', '_')
        except Exception:
            sym = "SYMBOL"
        try:
            st = (str(strat_name or 'STRAT')[:8]).upper().replace('|', '_')
        except Exception:
            st = "STRAT"
        try:
            suffix = str(idem_suffix) if idem_suffix else str(int(time.time()))
        except Exception:
            suffix = "0"
        coid = f"TB|{side}|{sym}|{st}|{suffix}"
        return coid[:48]

    def _make_idem_suffix(self, symbol, strat_name, side='B', price=None):
        """Create a deterministic short idempotency suffix for client_order_id.

        Uses the *last bar timestamp* (minute bucket) when available, else current time.
        Includes a rounded price component to reduce collisions across different signals.
        """
        try:
            sym = str(symbol).upper().replace('/', '_')
        except Exception:
            sym = "SYMBOL"
        try:
            st = (str(strat_name or 'STRAT')[:8]).upper().replace('|', '_')
        except Exception:
            st = "STRAT"

        try:
            bar_epoch = int((getattr(self, '_last_bar_epoch', {}) or {}).get(sym, 0) or 0)
        except Exception:
            bar_epoch = 0
        if bar_epoch <= 0:
            try:
                bar_epoch = int(time.time())
            except Exception:
                bar_epoch = 0

        bucket = int(bar_epoch // 60)

        p = 0
        if price is not None:
            try:
                p = int(round(float(price) * 100.0))
            except Exception:
                p = 0

        raw = f"{bucket}|{side}|{sym}|{st}|{p}"
        try:
            return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10].upper()
        except Exception:
            try:
                return hex(abs(hash(raw)))[2:12].upper()
            except Exception:
                return str(bucket)

    def _get_order_by_client_order_id_safe(self, client_order_id):
        """Best-effort lookup for an order by client_order_id (Alpaca support varies by SDK version)."""
        if not self.api or not client_order_id:
            return None
        try:
            if hasattr(self.api, 'get_order_by_client_order_id'):
                return self.retry_api_call(self.api.get_order_by_client_order_id, client_order_id)
        except Exception:
            pass
        # Fallback: scan recent orders
        try:
            orders = self.retry_api_call(self.api.list_orders, status='all', limit=500)
            for o in orders or []:
                try:
                    if str(getattr(o, 'client_order_id', '') or '') == str(client_order_id):
                        return o
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _adopt_open_buy_order(self, o):
        """Ensure a broker open BUY order is tracked in pending_orders (idempotency adopt)."""
        try:
            oid = getattr(o, 'id', None)
            if not oid:
                return
            oid = str(oid)
            if oid in self.pending_orders:
                return

            side = str(getattr(o, 'side', '')).lower()
            if side != 'buy':
                return

            sym = str(getattr(o, 'symbol', '')).upper()
            qty = float(getattr(o, 'qty', 0) or 0)
            limit_price = getattr(o, 'limit_price', None)
            try:
                limit_price = float(limit_price) if limit_price is not None else 0.0
            except Exception:
                limit_price = 0.0

            coid = getattr(o, 'client_order_id', '') or ''
            strat = None
            try:
                parts = str(coid).split('|')
                if len(parts) >= 4:
                    strat = parts[3]
            except Exception:
                strat = None

            created_at = getattr(o, 'created_at', None)
            try:
                created_ts = pd.to_datetime(created_at).timestamp() if created_at else time.time()
            except Exception:
                created_ts = time.time()

            self.pending_orders[oid] = {
                'symbol': sym,
                'qty': qty,
                'price': limit_price,
                'strategy': strat or 'UNKNOWN',
                'submitted_ts': created_ts,
                'order_class': getattr(o, 'order_class', None),
                'client_order_id': str(coid)[:48],
            }
            self._pending_symbols.add(sym)
        except Exception:
            return

    def _find_existing_open_entry_order(self, symbol):
        """Find an existing TradingBot-tagged open BUY order for symbol and adopt it if found."""
        if not self.api:
            return None
        try:
            sym = str(symbol).upper()
        except Exception:
            sym = symbol
        try:
            orders = self.retry_api_call(self.api.list_orders, status='open', limit=500)
        except Exception:
            return None
        for o in orders or []:
            try:
                if str(getattr(o, 'symbol', '')).upper() != sym:
                    continue
                if str(getattr(o, 'side', '')).lower() != 'buy':
                    continue
                coid = str(getattr(o, 'client_order_id', '') or '')
                if not coid.startswith('TB|'):
                    continue
                self._adopt_open_buy_order(o)
                return o
            except Exception:
                continue
        return None

    def _submit_order_idempotent(self, *, client_order_id: str, submit_kwargs: dict):
        """Submit an order with idempotency semantics using client_order_id.

        - Avoids retry loops that can duplicate orders after transient network failures.
        - If the broker already has an order with this client_order_id, returns it.
        """
        if not self.api:
            raise RuntimeError("API not connected")

        # Attempt 1: submit
        try:
            return self.api.submit_order(**submit_kwargs)
        except Exception as e:
            msg = str(e).lower()

            # Duplicate/exists: resolve by lookup
            if 'client_order_id' in msg and ('already' in msg or 'exists' in msg or 'duplicate' in msg):
                found = self._get_order_by_client_order_id_safe(client_order_id)
                if found is not None:
                    return found

            # Network-ish: check if order actually landed before retrying
            netish = isinstance(e, (ConnectionResetError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout))
            if netish or 'connection' in msg or 'read timed out' in msg or 'timeout' in msg:
                found = self._get_order_by_client_order_id_safe(client_order_id)
                if found is not None:
                    return found
                # One controlled retry with the same client_order_id
                try:
                    return self.api.submit_order(**submit_kwargs)
                except Exception as e2:
                    msg2 = str(e2).lower()
                    if 'client_order_id' in msg2 and ('already' in msg2 or 'exists' in msg2 or 'duplicate' in msg2):
                        found = self._get_order_by_client_order_id_safe(client_order_id)
                        if found is not None:
                            return found
                    raise
            raise

    def _close_position_idempotent(self, symbol: str) -> None:
        """Close a position with idempotency semantics (safe on retries)."""
        if not self.api:
            raise RuntimeError("API not connected")

        try:
            self.api.close_position(symbol)
            return
        except Exception as e:
            msg = str(e).lower()
            netish = isinstance(e, (ConnectionResetError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout))
            if netish or 'connection' in msg or 'read timed out' in msg or 'timeout' in msg:
                # If the position is already gone, treat as success
                try:
                    self.api.get_position(symbol)
                except Exception:
                    return
                # Otherwise retry once
                self.api.close_position(symbol)
                return
            if 'position does not exist' in msg or 'no position' in msg:
                return
            raise

    def reconcile_broker_state(self, force: bool = False):
        """Reconcile broker state vs internal DB/pending state (v5.12.5).

        Detects and logs state drift:
          - broker position exists but DB doesn't
          - DB shows position but broker doesn't (confirmed)
          - pending order missing from broker
        Optionally halts new entries when mismatches are detected.
        """
        if not self.api:
            return

        now_ts = time.time()
        try:
            interval = max(10, int(getattr(self, '_reconcile_interval_sec', 60) or 60))
        except Exception:
            interval = 60
        try:
            if not force and (now_ts - float(getattr(self, '_reconcile_last_ts', 0.0) or 0.0)) < interval:
                return
        except Exception:
            pass
        self._reconcile_last_ts = now_ts

        # Lazy init counters
        if not hasattr(self, '_recon_pos_missing_counts') or self._recon_pos_missing_counts is None:
            self._recon_pos_missing_counts = {}
        if not hasattr(self, '_recon_order_missing_counts') or self._recon_order_missing_counts is None:
            self._recon_order_missing_counts = {}

        # Fetch broker state
        try:
            positions = self.retry_api_call(self.api.list_positions) or []
        except Exception as e:
            self._emit(f"[E_RECON_API_POS] reconcile positions failed: {self._redact(e)}", level="WARN", category="SYNC", throttle_key="recon_pos_fail", throttle_sec=60)
            return
        try:
            open_orders = self.retry_api_call(self.api.list_orders, status='open', limit=500) or []
        except Exception:
            open_orders = []

        broker_pos = {}
        for p in positions:
            try:
                sym = str(getattr(p, 'symbol', '')).upper()
                broker_pos[sym] = p
            except Exception:
                continue

        broker_open_ids = set()
        broker_open_buy_syms = set()
        for o in open_orders:
            try:
                oid = str(getattr(o, 'id', '') or '')
                if oid:
                    broker_open_ids.add(oid)
                if str(getattr(o, 'side', '')).lower() == 'buy':
                    sym = str(getattr(o, 'symbol', '')).upper()
                    coid = str(getattr(o, 'client_order_id', '') or '')
                    if coid.startswith('TB|'):
                        broker_open_buy_syms.add(sym)
                        # adopt if we missed it
                        if oid and oid not in (self.pending_orders or {}):
                            self._adopt_open_buy_order(o)
            except Exception:
                continue

        # Compare to DB
        try:
            db_trades = self.db.get_active_trades() or {}
        except Exception:
            db_trades = {}
        db_syms = {str(s).upper() for s in (db_trades.keys() if hasattr(db_trades, 'keys') else [])}
        broker_syms = set(broker_pos.keys())

        # Broker has position but DB doesn't -> import
        missing_in_db = sorted(list(broker_syms - db_syms))
        for sym in missing_in_db:
            try:
                p = broker_pos.get(sym)
                qty = float(getattr(p, 'qty', 0) or 0)
                avg = float(getattr(p, 'avg_entry_price', 0) or 0)
                self.db.log_trade_entry(sym, qty, avg, 'IMPORTED_RECON')
                self._emit(f"[E_RECON_IMPORT] Imported broker position into DB: {sym} qty={qty} avg={avg}", level="WARN", category="SYNC", symbol=sym, throttle_key=f"recon_import_{sym}", throttle_sec=60)
            except Exception as e:
                self._emit(f"[E_RECON_IMPORT_FAIL] {sym}: {self._redact(e)}", level="WARN", category="SYNC", symbol=sym, throttle_key=f"recon_import_fail_{sym}", throttle_sec=60)

        # DB has position but broker doesn't -> confirm before deleting (avoid transient API gaps)
        missing_on_broker = sorted(list(db_syms - broker_syms))
        confirm_n = max(1, int(getattr(self, '_reconcile_confirmations', 2) or 2))
        for sym in missing_on_broker:
            try:
                c = int(self._recon_pos_missing_counts.get(sym, 0) or 0) + 1
                self._recon_pos_missing_counts[sym] = c
                if c >= confirm_n:
                    try:
                        self.db.remove_active_trade(sym)
                    except Exception:
                        pass
                    self._emit(f"[E_RECON_GHOST_DB] Removed DB ghost trade (no broker position): {sym}", level="ERROR", category="SYNC", symbol=sym, throttle_key=f"recon_ghost_{sym}", throttle_sec=120)
                else:
                    self._emit(f"[E_RECON_DB_NO_BROKER] DB has trade but broker missing (confirm {c}/{confirm_n}): {sym}", level="WARN", category="SYNC", symbol=sym, throttle_key=f"recon_warn_{sym}", throttle_sec=60)
            except Exception:
                continue

        # Reset counts for symbols that are now present
        for sym in list(self._recon_pos_missing_counts.keys()):
            if sym in broker_syms:
                self._recon_pos_missing_counts[sym] = 0

        # Pending order drift: pending_orders oid not open anymore and no position
        for oid, meta in list((self.pending_orders or {}).items()):
            try:
                if str(oid) in broker_open_ids:
                    self._recon_order_missing_counts[str(oid)] = 0
                    continue
                sym = str(meta.get('symbol', '')).upper()
                # If position exists, pending will be cleaned by process_pending_orders
                if sym and sym in broker_syms:
                    self._recon_order_missing_counts[str(oid)] = 0
                    continue
                c = int(self._recon_order_missing_counts.get(str(oid), 0) or 0) + 1
                self._recon_order_missing_counts[str(oid)] = c
                if c >= confirm_n:
                    del self.pending_orders[str(oid)]
                    self._pending_symbols.discard(sym)
                    self._emit(f"[E_RECON_PENDING_MISSING] Dropped missing pending order {oid} for {sym}", level="WARN", category="ORDER", symbol=sym, throttle_key=f"recon_pend_{sym}", throttle_sec=60)
            except Exception:
                continue

        # Optional safe-mode: halt new entries on detected mismatch
        try:
            halt_on = bool(getattr(self, '_reconcile_halt_on_mismatch', True))
        except Exception:
            halt_on = True
        try:
            mismatches = (len(missing_in_db) > 0) or (len(missing_on_broker) > 0)
        except Exception:
            mismatches = False

        if halt_on and mismatches and not getattr(self, '_entries_disabled', False):
            reason = f"RECON_MISMATCH: broker_missing={len(missing_on_broker)} internal_missing={len(missing_in_db)}"
            self._entries_disabled = True
            self._entries_disabled_reason = reason
            try:
                self._emit(f"⛔ Safe-mode: new entries disabled | {reason}", level="ERROR", category="RISK", throttle_key="recon_halt", throttle_sec=60)
            except Exception:
                pass
            try:
                self.send_telegram(f"⛔ Safe-mode: new entries disabled | {reason}")
            except Exception:
                pass
    def sync_open_orders(self):
        """Rebuild pending_orders from Alpaca open orders (best-effort)."""
        if not self.api:
            return
        try:
            orders = self.retry_api_call(self.api.list_orders, status='open', limit=500)
        except Exception:
            return

        rebuilt = 0
        for o in orders or []:
            try:
                # Only reconstruct TradingBot-tagged orders
                coid = getattr(o, 'client_order_id', '') or ''
                if not str(coid).startswith('TB|'):
                    continue

                oid = getattr(o, 'id', None)
                if not oid or oid in self.pending_orders:
                    continue

                side = str(getattr(o, 'side', '')).lower()
                if side != 'buy':
                    continue

                sym = str(getattr(o, 'symbol', '')).upper()
                qty = float(getattr(o, 'qty', 0) or 0)
                limit_price = getattr(o, 'limit_price', None)
                try:
                    limit_price = float(limit_price) if limit_price is not None else 0.0
                except Exception:
                    limit_price = 0.0

                # Parse strategy from client_order_id if present
                strat = None
                try:
                    parts = str(coid).split('|')
                    # TB|B|SYM|STRAT|TS
                    if len(parts) >= 4:
                        strat = parts[3]
                except Exception:
                    strat = None

                created_at = getattr(o, 'created_at', None)
                try:
                    created_ts = pd.to_datetime(created_at).timestamp() if created_at else time.time()
                except Exception:
                    created_ts = time.time()

                self.pending_orders[str(oid)] = {
                    'symbol': sym,
                    'qty': qty,
                    'price': limit_price,
                    'strategy': strat or 'UNKNOWN',
                    'submitted_ts': created_ts,
                    'order_class': getattr(o, 'order_class', None),
                }
                self._pending_symbols.add(sym)
                rebuilt += 1
            except Exception:
                continue

        if rebuilt and self._log_order_lifecycle:
            self._emit(f"🧾 Rebuilt {rebuilt} pending order(s) from Alpaca.", level="INFO", category="ORDER")

    def process_pending_orders(self):
        """Poll pending buy orders; log DB entry only when filled; cancel if TTL exceeded."""
        if not self.pending_orders or not self.api:
            return

        now_ts = time.time()
        ttl = float(self._live_entry_ttl_sec or 0)

        for oid, meta in list(self.pending_orders.items()):
            try:
                sym = str(meta.get('symbol', '')).upper()
                strat = meta.get('strategy', 'UNKNOWN')

                # If position already exists, treat as filled and write DB
                try:
                    pos = self.retry_api_call(self.api.get_position, sym)
                    if pos is not None:
                        filled_price = float(getattr(pos, 'avg_entry_price', meta.get('price', 0)) or meta.get('price', 0))
                        qty = float(getattr(pos, 'qty', meta.get('qty', 0)) or meta.get('qty', 0))
                        self.db.log_trade_entry(sym, qty, filled_price, strat)
                        self._log_exec_packet(symbol=sym, side="BUY", phase="FILL", decision_id=meta.get("decision_id"), qty=qty, price=filled_price, order_id=oid, client_order_id=meta.get("client_order_id"), broker_status="filled_via_position", payload={"strategy": strat})
                        try:
                            self._order_stats['filled'] = int(self._order_stats.get('filled', 0)) + 1
                        except Exception:
                            pass
                        self._emit(f"✅ FILLED {sym} | Qty {qty} @ ${filled_price:.2f} | {strat}", level="INFO", category="ORDER", symbol=sym)
                        del self.pending_orders[oid]
                        self._pending_symbols.discard(sym)
                        continue
                except Exception:
                    pass

                # Otherwise poll the order itself
                try:
                    o = self.retry_api_call(self.api.get_order, oid)
                except Exception:
                    o = None

                status = str(getattr(o, 'status', '')).lower() if o is not None else ''
                if status in ('filled',):
                    fp = getattr(o, 'filled_avg_price', None)
                    try:
                        fp = float(fp) if fp is not None else float(meta.get('price', 0))
                    except Exception:
                        fp = float(meta.get('price', 0) or 0)
                    try:
                        fq = float(getattr(o, 'filled_qty', None) or meta.get('qty', 0))
                    except Exception:
                        fq = float(meta.get('qty', 0) or 0)
                    self.db.log_trade_entry(sym, fq, fp, strat)
                    self._log_exec_packet(symbol=sym, side="BUY", phase="FILL", decision_id=meta.get("decision_id"), qty=fq, price=fp, order_id=oid, client_order_id=meta.get("client_order_id"), broker_status=status, payload={"strategy": strat})
                    try:
                        self._order_stats['filled'] = int(self._order_stats.get('filled', 0)) + 1
                    except Exception:
                        pass
                    self._emit(f"✅ FILLED {sym} | Qty {fq} @ ${fp:.2f} | {strat}", level="INFO", category="ORDER", symbol=sym)
                    del self.pending_orders[oid]
                    self._pending_symbols.discard(sym)
                    continue

                if status in ('canceled', 'rejected', 'expired'):
                    try:
                        key = status if status in ('canceled','rejected','expired') else 'canceled'
                        self._order_stats[key] = int(self._order_stats.get(key, 0)) + 1
                    except Exception:
                        pass
                    if status == 'rejected':
                        try:
                            self._log_exec_packet(symbol=sym, side="BUY", phase=status.upper(), decision_id=meta.get("decision_id"), qty=meta.get("qty"), price=meta.get("price"), order_id=oid, client_order_id=meta.get("client_order_id"), broker_status=status, payload={"strategy": strat})
                        except Exception:
                            pass
                        try:
                            self._e5_note_reject(sym)
                        except Exception:
                            pass
                    if self._log_order_lifecycle:
                        self._emit(f"❌ BUY {sym} {status.upper()} | {strat}", level="WARN", category="ORDER", symbol=sym)
                    del self.pending_orders[oid]
                    self._pending_symbols.discard(sym)
                    continue

                # TTL handling
                submitted_ts = float(meta.get('submitted_ts', now_ts) or now_ts)
                age = now_ts - submitted_ts
                if ttl and age > ttl and self._live_cancel_unfilled_entries:
                    try:
                        self._order_stats['ttl_canceled'] = int(self._order_stats.get('ttl_canceled', 0)) + 1
                        self._order_stats['canceled'] = int(self._order_stats.get('canceled', 0)) + 1
                    except Exception:
                        pass
                    try:
                        self._e5_note_ttl_cancel(sym)
                    except Exception:
                        pass
                    try:
                        self.retry_api_call(self.api.cancel_order, oid)
                    except Exception:
                        pass
                    try:
                        self._log_exec_packet(symbol=sym, side="BUY", phase="TTL_CANCEL", decision_id=meta.get("decision_id"), qty=meta.get("qty"), price=meta.get("price"), order_id=oid, client_order_id=meta.get("client_order_id"), broker_status="canceled", payload={"age_sec": age, "strategy": strat})
                    except Exception:
                        pass
                    if self._log_order_lifecycle:
                        self._emit(f"⏱️ CANCELED unfilled BUY {sym} after {int(age)}s | {strat}", level="WARN", category="ORDER", symbol=sym, throttle_key=f"ttl_{sym}", throttle_sec=30)
                    del self.pending_orders[oid]
                    self._pending_symbols.discard(sym)
            except Exception as e:
                self._emit(f"Order monitor error: {self._redact(e)}", level="WARN", category="ORDER", throttle_key="order_mon_err", throttle_sec=60)

    def cancel_all_pending_orders(self):
        """Best-effort cleanup for outstanding entry orders when stopping / kill-switching."""
        if not self.pending_orders or not self.api:
            return
        canceled = 0
        for oid, meta in list(self.pending_orders.items()):
            sym = str(meta.get('symbol', '')).upper()
            try:
                self.retry_api_call(self.api.cancel_order, oid)
            except Exception:
                pass
            try:
                self._log_exec_packet(symbol=sym, side="BUY", phase="CANCEL_ALL", decision_id=meta.get("decision_id"), qty=meta.get("qty"), price=meta.get("price"), order_id=oid, client_order_id=meta.get("client_order_id"), broker_status="canceled", payload={})
            except Exception:
                pass
            self._pending_symbols.discard(sym)
            del self.pending_orders[oid]
            canceled += 1
            try:
                self._order_stats['canceled'] = int(self._order_stats.get('canceled', 0)) + 1
            except Exception:
                pass
        if canceled and self._log_order_lifecycle:
            self._emit(f"🧹 Canceled {canceled} pending order(s).", level="INFO", category="ORDER")

    def send_telegram(self, msg):
        if self.config['KEYS'].get('telegram_enabled', 'True').lower() != 'true': return
        try:
            token = self.config['KEYS']['telegram_token'].strip()
            chat_id = self.config['KEYS']['telegram_chat_id'].strip()
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
        except Exception as e:
            # Do not risk logging tokens/URLs. Provide a minimal error for diagnostics.
            self._emit(f"⚠️ Telegram Failed: {type(e).__name__}", level="WARN", category="TELEGRAM", throttle_key="tg_fail", throttle_sec=300)

    def check_telegram_commands(self):
        if self.config['KEYS'].get('telegram_enabled', 'True').lower() != 'true': return
        try:
            token = self.config['KEYS']['telegram_token'].strip()
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {'offset': self.last_telegram_update_id + 1, 'timeout': 1}
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for res in data.get('result', []):
                    self.last_telegram_update_id = res['update_id']
                    if 'message' in res and 'text' in res['message']:
                        cmd = res['message']['text'].strip().lower()
                        self.process_command(cmd)
        except Exception as e:
            # v5.12.3 updateA: don't silently fail telegram polling.
            # Avoid logging any token/URL details.
            try:
                self._emit(
                    f"⚠️ [E_TG_UPDATES_FAIL] Telegram polling failed: {type(e).__name__}",
                    level="WARN",
                    category="TELEGRAM",
                    throttle_key="tg_updates_fail",
                    throttle_sec=300,
                )
            except Exception:
                pass

    def process_command(self, cmd):
        self.log(f"📩 Telegram Command: {cmd}")
        if cmd == "/status":
            self.send_telegram(f"STATUS: {self.market_status}\nEquity: ${self.current_equity:.2f}\nActive Trades: {len(self.db.get_active_trades())}")
        elif cmd == "/liquidate":
            self.send_telegram("⚠️ PANIC SELL INITIATED!")
            self.liquidate_all()
        elif cmd.startswith("/buy "):
            symbol = cmd.replace("/buy ", "").upper().strip()
            self.log(f"⚠️ Manual Force Buy: {symbol}")
            self.send_telegram(f"⏳ Attempting to buy {symbol}...")
            try:
                if '/' in symbol:
                    bar = self.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                else:
                    bar = self.retry_api_call(self.api.get_latest_bar, symbol)
                
                price = bar['close'] if isinstance(bar, pd.Series) else bar.c
                allowed, reason = self.wallet.can_buy(symbol, price, self.current_equity)
                if allowed:
                    if symbol in self._pending_symbols:
                        self.send_telegram(f"⏳ Pending order already open for {symbol}.")
                        return
                    qty = self.wallet.get_trade_qty(price, equity=self.current_equity)
                    self.execute_buy(symbol, qty, price, "MANUAL_OVERRIDE", 0.0)
                else:
                    self.send_telegram(f"❌ Denied: {reason}")
            except Exception as e:
                 self.send_telegram(f"❌ Error: {e}")
        else:
            self.send_telegram("❓ Unknown Command.\nTry: /status, /liquidate, /buy SYMBOL")
    def stop(self, reason="USER_STOP"):
        try:
            cur = getattr(self, '_stop_reason', None)
            if (cur is None) or (cur == 'RUNNING'):
                self._stop_reason = reason
        except Exception:
            self._stop_reason = reason

        self.active = False
        # Release D1: avoid leaving dangling entry orders
        try:
            self.cancel_all_pending_orders()
        except Exception:
            pass
        self._emit("🛑 Stopping Engine...", category="SYSTEM")
        self._publish_agent_event("ENGINE_STOPPED", {"reason": self._stop_reason})


    # --------------------
    # Release E3: End-of-run summary export
    # --------------------
    def export_run_summary(self, force: bool = False):
        """Write a JSON run summary into the logs directory.

        Triggered automatically when the engine stops.
        """
        if not getattr(self, '_run_summary_enabled', True) and not force:
            return None
        if getattr(self, '_session_started_at', None) is None and not force:
            return None
        if getattr(self, '_session_started_at', None) is None and force:
            # Allow manual export even if a run has not been started
            self._session_started_at = datetime.now()
            if getattr(self, '_session_id', None) is None:
                try:
                    self._session_id = self._session_started_at.strftime('%Y.%m.%d_%H.%M.%S')
                except Exception:
                    self._session_id = None

        ended_at = datetime.now()
        started_at = getattr(self, '_session_started_at', ended_at)
        try:
            duration_sec = max(0.0, (ended_at - started_at).total_seconds())
        except Exception:
            duration_sec = 0.0

        # Pull session-scoped trades/decisions
        try:
            trades = self.db.get_trade_history_since(started_at) or []
        except Exception:
            trades = []
        try:
            decisions = self.db.get_decision_counts_since(started_at) or {}
        except Exception:
            decisions = {}

        pls = []
        by_strategy = {}
        try:
            for r in trades:
                try:
                    pl = float(r.get('profit_loss', 0.0) or 0.0)
                    pls.append(pl)
                except Exception:
                    pass
                try:
                    s = str(r.get('strategy', 'UNKNOWN') or 'UNKNOWN')
                    by_strategy.setdefault(s, {'trades': 0, 'total_pl': 0.0})
                    by_strategy[s]['trades'] += 1
                    by_strategy[s]['total_pl'] += float(r.get('profit_loss', 0.0) or 0.0)
                except Exception:
                    pass
        except Exception:
            pass

        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        wins = [x for x in pls if x > 0]
        losses = [x for x in pls if x < 0]
        total_pl = float(sum(pls)) if pls else 0.0
        win_rate = float(len(wins) / len(pls)) if pls else 0.0
        avg_win = _mean(wins)
        avg_loss = _mean(losses)
        profit_factor = None
        try:
            loss_sum = abs(sum(losses))
            profit_factor = (sum(wins) / loss_sum) if loss_sum > 0 else (float('inf') if wins else 0.0)
        except Exception:
            profit_factor = None
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

        # Max drawdown on cumulative P/L
        max_dd = 0.0
        try:
            peak = 0.0
            cum = 0.0
            for x in pls:
                cum += float(x)
                peak = max(peak, cum)
                dd = peak - cum
                max_dd = max(max_dd, dd)
        except Exception:
            max_dd = 0.0

        # Data freshness via DB last bar timestamps
        stale_info = {
            'threshold_seconds': int(getattr(self, '_stale_bar_seconds_threshold', 0) or 0),
            'stale_symbols': [],
            'symbol_age_seconds': {},
            'oldest_age_seconds': None,
        }
        try:
            threshold = float(getattr(self, '_stale_bar_seconds_threshold', 0) or 0)
            syms = []
            try:
                from ..watchlist_api import get_watchlist_symbols
                syms.extend([s.upper() for s in get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL")])
            except Exception:
                pass
            syms = sorted(set([s for s in syms if s]))
            latest = self.db.get_latest_timestamps_for_symbols(syms) if syms else {}
            now_dt = datetime.now(timezone.utc)
            oldest = None
            for sym, ts in (latest or {}).items():
                try:
                    if ts is None:
                        continue
                    dt = pd.to_datetime(ts, utc=True).to_pydatetime()
                    age = float((now_dt - dt).total_seconds())
                    stale_info['symbol_age_seconds'][str(sym).upper()] = age
                    if oldest is None or age > oldest:
                        oldest = age
                    if threshold > 0 and age > threshold:
                        stale_info['stale_symbols'].append(str(sym).upper())
                except Exception:
                    pass
            if oldest is not None:
                stale_info['oldest_age_seconds'] = float(oldest)
        except Exception:
            pass

        # AI training snapshot (best-effort)
        try:
            ai_stats = getattr(self.ai, 'last_train_stats', {}) or {}
        except Exception:
            ai_stats = {}

        summary = {
            'engine': {
                'version': APP_VERSION,
                'release': APP_RELEASE,
                'session_id': getattr(self, '_session_id', None),
                'started_at': started_at,
                'ended_at': ended_at,
                'duration_seconds': duration_sec,
                'stop_reason': getattr(self, '_stop_reason', None),
            },
            'equity': {
                'start_equity': float(getattr(self, 'start_equity', 0.0) or 0.0),
                'end_equity': float(getattr(self, 'current_equity', 0.0) or 0.0),
                'pl_dollars': float(getattr(self, 'current_equity', 0.0) or 0.0) - float(getattr(self, 'start_equity', 0.0) or 0.0),
            },
            'order_lifecycle': dict(getattr(self, '_order_stats', {}) or {}),
            'trades_session': {
                'count': int(len(pls)),
                'wins': int(len(wins)),
                'losses': int(len(losses)),
                'win_rate': win_rate,
                'total_pl': total_pl,
                'avg_pl': _mean(pls),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'max_drawdown_pl': float(max_dd),
                'by_strategy': by_strategy,
            },
            'decisions_session': decisions,
            'data_freshness': stale_info,
            'ai': ai_stats,
            'watchdog': {
                'e5_enabled': bool(getattr(self, '_e5_enabled', True)),
                'enforced': bool(self._e5_is_enforced()) if hasattr(self, '_e5_is_enforced') else False,
                'entries_disabled': bool(getattr(self, '_entries_disabled', False)),
                'entries_disabled_reason': getattr(self, '_entries_disabled_reason', None),
                'window_seconds': int(getattr(self, '_e5_event_window_sec', 0) or 0),
                'api_errors_in_window': int(len(getattr(self, '_e5_api_error_events', []) or [])),
                'rejects_in_window': int(len(getattr(self, '_e5_reject_events', []) or [])),
                'ttl_cancels_in_window': int(len(getattr(self, '_e5_ttl_cancel_events', []) or [])),
            },
        }

        try:
            logs_dir = None
            try:
                logs_dir = (getattr(self, '_paths', None) or {}).get('logs')
            except Exception:
                logs_dir = None
            if not logs_dir:
                logs_dir = (get_paths() or {}).get('logs')
            if not logs_dir:
                logs_dir = os.path.abspath('.')
            os.makedirs(logs_dir, exist_ok=True)

            fname_ts = ended_at.strftime('%Y.%m.%d_%H.%M.%S')
            
            # Determine session tag from Alpaca base_url (paper vs live)
            session_tag = 'paper'
            try:
                base_url = ''
                try:
                    base_url = str((self.config.get('KEYS', {}) or {}).get('base_url', '') or '').lower()
                except Exception:
                    base_url = ''
                if base_url and ('paper' not in base_url):
                    session_tag = 'live'
            except Exception:
                session_tag = 'paper'
            # Persist the session tag into the exported JSON as well
            try:
                if isinstance(summary.get('engine', None), dict):
                    summary['engine']['session_tag'] = session_tag
                else:
                    summary['session_tag'] = session_tag
            except Exception:
                pass

            
            summaries_dir = os.path.join(logs_dir, 'summaries')
            os.makedirs(summaries_dir, exist_ok=True)
            # --------------------
            # Promotion status (paper → live readiness) - reporting only
            # --------------------
            try:
                cfg = {}
                try:
                    cfg = (self.config.get('CONFIGURATION', {}) or {})
                except Exception:
                    cfg = {}

                def _cfg_get(name, default=None):
                    try:
                        v = cfg.get(name, default)
                        return default if v is None else v
                    except Exception:
                        return default

                def _to_bool(v, default=False):
                    if isinstance(v, bool):
                        return v
                    if v is None:
                        return default
                    s = str(v).strip().lower()
                    if s in ('1', 'true', 'yes', 'y', 'on'):
                        return True
                    if s in ('0', 'false', 'no', 'n', 'off'):
                        return False
                    return default

                def _to_int(v, default=0):
                    try:
                        return int(float(str(v).strip()))
                    except Exception:
                        return default

                def _to_float(v, default=0.0):
                    try:
                        return float(str(v).strip())
                    except Exception:
                        return default

                promo_enabled = _to_bool(_cfg_get('promotion_enabled', True), True)
                promo_min_sessions = max(0, _to_int(_cfg_get('promotion_min_sessions', 5), 5))
                promo_min_trades_total = max(0, _to_int(_cfg_get('promotion_min_trades_total', 30), 30))
                promo_max_dd_pct = _to_float(_cfg_get('promotion_max_drawdown_pct', 4.0), 4.0)
                promo_max_daily_loss_pct = _to_float(_cfg_get('promotion_max_daily_loss_pct', 2.0), 2.0)
                promo_max_cancel_rate_pct = _to_float(_cfg_get('promotion_max_cancel_rate_pct', 35.0), 35.0)
                promo_max_reject_rate_pct = _to_float(_cfg_get('promotion_max_reject_rate_pct', 10.0), 10.0)
                promo_max_api_errors_per_hour = _to_int(_cfg_get('promotion_max_api_errors_per_hour', 20), 20)
                promo_require_no_stale = _to_bool(_cfg_get('promotion_require_no_stale_symbols', True), True)
                promo_require_no_crashes = _to_bool(_cfg_get('promotion_require_no_crashes', True), True)
                promo_require_no_watchdog_halts = _to_bool(_cfg_get('promotion_require_no_watchdog_halts', True), True)
                promo_window_days = max(1, _to_int(_cfg_get('promotion_window_days', 30), 30))

                # Collect recent paper summaries from disk (and include current summary if paper)
                import glob
                import re as _re
                import os as _os

                re_pat = _re.compile(r'^summary_(\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2})_(paper|live)\.json$')
                now_local = ended_at
                cutoff = now_local - timedelta(days=promo_window_days)

                sessions = []
                try:
                    for fp in glob.glob(_os.path.join(summaries_dir, 'summary_*_paper.json')):
                        bn = _os.path.basename(fp)
                        m2 = re_pat.match(bn)
                        if not m2:
                            continue
                        ts_str = m2.group(1)
                        try:
                            dt = datetime.strptime(ts_str, '%Y.%m.%d_%H.%M.%S')
                        except Exception:
                            dt = None
                        if dt and dt < cutoff:
                            continue
                        try:
                            with open(fp, 'r', encoding='utf-8') as _f:
                                sessions.append(json.load(_f))
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    cur_tag = None
                    try:
                        cur_tag = (summary.get('engine', {}) or {}).get('session_tag', None)
                    except Exception:
                        cur_tag = None
                    if str(cur_tag).lower() == 'paper':
                        sessions.append(summary)
                except Exception:
                    pass

                def _parse_ended(s):
                    try:
                        v = (s.get('engine', {}) or {}).get('ended_at', None)
                        if v is None:
                            return None
                        return pd.to_datetime(v).to_pydatetime()
                    except Exception:
                        return None

                try:
                    sessions_sorted = sorted(sessions, key=lambda s: (_parse_ended(s) or datetime.min))
                except Exception:
                    sessions_sorted = sessions

                # Consider most recent N sessions (where N=min_sessions)
                recent = sessions_sorted[-promo_min_sessions:] if promo_min_sessions > 0 else sessions_sorted

                def _is_market_hours(dt):
                    try:
                        if dt is None:
                            return False
                        if dt.weekday() >= 5:
                            return False
                        t = dt.time()
                        return (t >= datetime.strptime('08:30', '%H:%M').time() and t <= datetime.strptime('15:00', '%H:%M').time())
                    except Exception:
                        return False

                total_trades = 0
                total_submitted = 0
                total_canceled = 0
                total_rejected = 0
                worst_dd_pct = 0.0
                worst_loss_pct = 0.0
                max_api_err_per_hour = 0.0
                stale_sessions = 0
                watchdog_halt_sessions = 0
                crash_sessions = 0

                for s in recent:
                    try:
                        total_trades += int((s.get('trades_session', {}) or {}).get('count', 0) or 0)
                    except Exception:
                        pass
                    try:
                        eq = (s.get('equity', {}) or {})
                        start_eq = float(eq.get('start_equity', 0.0) or 0.0)
                        pl_d = float(eq.get('pl_dollars', 0.0) or 0.0)
                        dd_pl = float((s.get('trades_session', {}) or {}).get('max_drawdown_pl', 0.0) or 0.0)
                        if start_eq > 0:
                            dd_pct = float(dd_pl) / start_eq * 100.0
                            worst_dd_pct = max(worst_dd_pct, dd_pct)
                            loss_pct = (abs(pl_d) / start_eq * 100.0) if pl_d < 0 else 0.0
                            worst_loss_pct = max(worst_loss_pct, loss_pct)
                    except Exception:
                        pass
                    try:
                        ol = (s.get('order_lifecycle', {}) or {})
                        submitted = int(ol.get('submitted', 0) or 0)
                        canceled = int(ol.get('canceled', 0) or 0)
                        ttl_c = int(ol.get('ttl_canceled', 0) or 0)
                        rejected = int(ol.get('rejected', 0) or 0)
                        total_submitted += submitted
                        total_canceled += (canceled + ttl_c)
                        total_rejected += rejected
                    except Exception:
                        pass
                    try:
                        wd = (s.get('watchdog', {}) or {})
                        wsec = int(wd.get('window_seconds', 0) or 0)
                        api_e = int(wd.get('api_errors_in_window', 0) or 0)
                        if wsec > 0:
                            max_api_err_per_hour = max(max_api_err_per_hour, (api_e * 3600.0) / float(wsec))
                        if bool(wd.get('entries_disabled', False)):
                            watchdog_halt_sessions += 1
                    except Exception:
                        pass
                    try:
                        dt_end = _parse_ended(s)
                        if _is_market_hours(dt_end):
                            stales = (s.get('data_freshness', {}) or {}).get('stale_symbols', []) or []
                            if len(stales) > 0:
                                stale_sessions += 1
                    except Exception:
                        pass
                    try:
                        if promo_require_no_crashes:
                            sr = str((s.get('engine', {}) or {}).get('stop_reason', '') or '').lower()
                            if any(x in sr for x in ['exception', 'traceback', 'crash', 'fatal']):
                                crash_sessions += 1
                    except Exception:
                        pass

                cancel_rate_pct = (float(total_canceled) / float(total_submitted) * 100.0) if total_submitted > 0 else None
                reject_rate_pct = (float(total_rejected) / float(total_submitted) * 100.0) if total_submitted > 0 else None

                failed = []
                notes = []

                if not promo_enabled:
                    failed.append('promotion_disabled')
                if promo_min_sessions > 0 and len(recent) < promo_min_sessions:
                    failed.append('min_sessions_not_met')
                if promo_min_trades_total > 0 and total_trades < promo_min_trades_total:
                    failed.append('min_trades_total_not_met')
                if worst_dd_pct > promo_max_dd_pct:
                    failed.append('max_drawdown_exceeded')
                if worst_loss_pct > promo_max_daily_loss_pct:
                    failed.append('max_daily_loss_exceeded')
                if cancel_rate_pct is not None and cancel_rate_pct > promo_max_cancel_rate_pct:
                    failed.append('cancel_rate_exceeded')
                if cancel_rate_pct is None:
                    notes.append('cancel_rate_not_computed_no_submitted_orders')
                if reject_rate_pct is not None and reject_rate_pct > promo_max_reject_rate_pct:
                    failed.append('reject_rate_exceeded')
                if reject_rate_pct is None:
                    notes.append('reject_rate_not_computed_no_submitted_orders')
                if max_api_err_per_hour > float(promo_max_api_errors_per_hour):
                    failed.append('api_errors_per_hour_exceeded')
                if promo_require_no_stale and stale_sessions > 0:
                    failed.append('stale_symbols_detected_during_market_hours')
                if promo_require_no_watchdog_halts and watchdog_halt_sessions > 0:
                    failed.append('watchdog_halts_detected')
                if promo_require_no_crashes and crash_sessions > 0:
                    failed.append('crash_detected')

                eligible = (len(failed) == 0)

                summary['promotion_status'] = {
                    'enabled': bool(promo_enabled),
                    'eligible': bool(eligible),
                    'checked_sessions': int(len(recent)),
                    'window_days': int(promo_window_days),
                    'requirements': {
                        'min_sessions': int(promo_min_sessions),
                        'min_trades_total': int(promo_min_trades_total),
                        'max_drawdown_pct': float(promo_max_dd_pct),
                        'max_daily_loss_pct': float(promo_max_daily_loss_pct),
                        'max_cancel_rate_pct': float(promo_max_cancel_rate_pct),
                        'max_reject_rate_pct': float(promo_max_reject_rate_pct),
                        'max_api_errors_per_hour': int(promo_max_api_errors_per_hour),
                        'require_no_stale_symbols': bool(promo_require_no_stale),
                        'require_no_watchdog_halts': bool(promo_require_no_watchdog_halts),
                        'require_no_crashes': bool(promo_require_no_crashes),
                        'market_hours_local': '08:30-15:00 America/Chicago',
                    },
                    'computed_metrics': {
                        'total_trades': int(total_trades),
                        'worst_drawdown_pct': float(worst_dd_pct),
                        'worst_session_loss_pct': float(worst_loss_pct),
                        'cancel_rate_pct': cancel_rate_pct,
                        'reject_rate_pct': reject_rate_pct,
                        'max_api_errors_per_hour': float(max_api_err_per_hour),
                        'stale_sessions_during_market_hours': int(stale_sessions),
                        'watchdog_halt_sessions': int(watchdog_halt_sessions),
                        'crash_sessions': int(crash_sessions),
                        'orders_submitted_total': int(total_submitted),
                    },
                    'failed_checks': failed,
                    'notes': notes,
                }
            except Exception:
                pass
            
            fname = f"summary_{fname_ts}_{session_tag}.json"
            out_path = os.path.join(summaries_dir, fname)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            self._emit(f"📄 Run summary written: {fname}", category='SYSTEM')
            return out_path
        except Exception as e:
            try:
                self._emit(f"⚠️ Summary write failed: {e}", level='WARN', category='SYSTEM')
            except Exception:
                pass
            return None

    def export_strategy_selection_report(self, force: bool = False):
        """
        Export a strategy selection report based on the latest backtest_results table.

        Output:
          logs/research/strategy_report_YYYY.MM.DD_HH.MM.SS_{paper|live}.json
          logs/research/strategy_report_YYYY.MM.DD_HH.MM.SS_{paper|live}.csv

        Notes:
          - This does NOT change trading logic.
          - If backtest_results is empty, we log a warning and optionally still write a stub JSON when force=True.
        """
        try:
            paths = get_paths()
            logs_dir = paths.get('logs') or os.path.join(paths.get('root', '.'), 'logs')
            out_dir = os.path.join(logs_dir, 'research')
            os.makedirs(out_dir, exist_ok=True)

            base_url = ''
            try:
                base_url = (self.config.get('KEYS', {}).get('alpaca_base_url', '') or '').lower()
            except Exception:
                base_url = ''
            session_tag = 'paper' if 'paper' in base_url else 'live'

            ts = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
            json_path = os.path.join(out_dir, f"strategy_report_{ts}_{session_tag}.json")
            csv_path = os.path.join(out_dir, f"strategy_report_{ts}_{session_tag}.csv")

            df = None
            try:
                df = self.db.get_backtest_data()
            except Exception:
                df = None

            if df is None or getattr(df, 'empty', True):
                msg = "[RESEARCH] ⚠️ Backtest results are empty. Run a full backtest first, then export again."
                self.log(msg)
                if force:
                    payload = {
                        "type": "strategy_selection_report",
                        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "session_tag": session_tag,
                        "status": "empty_backtest_results",
                        "note": "backtest_results table was empty at export time",
                    }
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2)
                    return json_path
                return None

            # Expect schema:
            # symbol | PL_<strategy>... | Trades_<strategy>... | best_strategy | best_profit
            df = df.copy()

            # Normalize columns
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str)
            if 'best_strategy' in df.columns:
                df['best_strategy'] = df['best_strategy'].astype(str)
            if 'best_profit' in df.columns:
                df['best_profit'] = pd.to_numeric(df['best_profit'], errors='coerce').fillna(0.0)

            # Collect strategy columns
            pl_cols = [c for c in df.columns if c.startswith('PL_')]
            trades_cols = [c for c in df.columns if c.startswith('Trades_')]
            strategies = sorted({c.replace('PL_', '') for c in pl_cols})

            # Build per-strategy aggregates from best_strategy choice
            best_counts = df['best_strategy'].value_counts(dropna=False).to_dict() if 'best_strategy' in df.columns else {}
            agg_by_strategy = {}
            for s in strategies:
                # how often strategy was selected as best
                count = int(best_counts.get(s, 0))
                # sum of best profits for symbols where selected
                if 'best_strategy' in df.columns:
                    mask = df['best_strategy'] == s
                    total_best_profit = float(df.loc[mask, 'best_profit'].sum())
                    mean_best_profit = float(df.loc[mask, 'best_profit'].mean()) if mask.any() else 0.0
                    win_rate = float((df.loc[mask, 'best_profit'] > 0).mean()) if mask.any() else 0.0
                else:
                    total_best_profit = 0.0
                    mean_best_profit = 0.0
                    win_rate = 0.0
                agg_by_strategy[s] = {
                    "symbols_best": count,
                    "total_best_profit": total_best_profit,
                    "mean_best_profit": mean_best_profit,
                    "win_rate_best": win_rate,
                }

            # Export CSV (one row per symbol)
            try:
                df.to_csv(csv_path, index=False)
            except Exception as e:
                self.log(f"[RESEARCH] ⚠️ Could not write CSV report: {e}")

            payload = {
                "type": "strategy_selection_report",
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "session_tag": session_tag,
                "rows": int(len(df)),
                "strategies": strategies,
                "columns": list(df.columns),
                "by_strategy": agg_by_strategy,
                "notes": [
                    "Report is generated from backtest_results (per-symbol best strategy snapshot).",
                    "For deeper evaluation, run multiple backtest windows and compare outputs over time.",
                ],
                "paths": {
                    "csv": os.path.basename(csv_path),
                    "json": os.path.basename(json_path),
                },
            }

            # Add compact per-symbol summary
            per_symbol = []
            for _, row in df.iterrows():
                item = {
                    "symbol": str(row.get('symbol', '')),
                    "best_strategy": str(row.get('best_strategy', '')),
                    "best_profit": float(row.get('best_profit', 0.0)),
                }
                # include full strategy PL/trades maps (compact)
                pl_map = {}
                tr_map = {}
                for s in strategies:
                    pl_map[s] = float(pd.to_numeric(row.get(f'PL_{s}', 0.0), errors='coerce') or 0.0)
                    tr_map[s] = int(pd.to_numeric(row.get(f'Trades_{s}', 0), errors='coerce') or 0)
                item["strategy_pl"] = pl_map
                item["strategy_trades"] = tr_map
                per_symbol.append(item)
            payload["per_symbol"] = per_symbol

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

            self.log(f"[RESEARCH] ✅ Strategy report exported: {json_path}")
            return json_path

        except Exception as e:
            self.log(f"[RESEARCH] ❌ Strategy report export failed: {e}")
            return None


    def reload_strategies(self):
        self.optimizer.load_strategies()
        self.wallet.reload_config()
        self._reload_logging_config()
        self._reload_execution_config()
        self._reload_e3_config()
        self._reload_e4_config()
        self._reload_e5_config()
        try:
            base_amt = float(self.config['CONFIGURATION'].get('amount_to_trade', 2000))
            base_loss = float(self.config['CONFIGURATION'].get('max_daily_loss', 100))
            self.kill_switch_ratio = base_loss / base_amt 
        except: self.kill_switch_ratio = 0.05
        
        self.connect_api()
        self.log("🔄 Config & Strategies Reloaded.")

    def run(self):
        if self.api is None: return
        self.active = True
        # Release E3: session bookkeeping
        try:
            self._session_started_at = datetime.now()
            self._session_id = self._session_started_at.strftime('%Y.%m.%d_%H.%M.%S')
            self._stop_reason = 'RUNNING'
            # reset counters each run
            try:
                for k in list(self._order_stats.keys()):
                    self._order_stats[k] = 0
            except Exception:
                self._order_stats = {'submitted': 0, 'filled': 0, 'canceled': 0, 'rejected': 0, 'expired': 0, 'ttl_canceled': 0}
            self._last_bar_epoch = {}
        except Exception:
            pass

        self._emit(f"🚀 {APP_VERSION} Engine Started ({APP_RELEASE})", category="SYSTEM")
        self._publish_agent_event("ENGINE_STARTED", {"session_id": self._session_id})
        self.send_telegram(f"🚀 TradingBot {APP_VERSION} Started. {APP_RELEASE} Mode Active.")
        self.sync_positions()
        self.check_market_regime()
        
        scan_counter = 0
        while self.active:
            self.check_kill_switch()
            try:
                self._e5_watchdog_tick()
            except Exception:
                pass
            if not self.active: break 
            
            self.check_telegram_commands()
            self.check_market_regime()
            # v5.12.5: periodic reconciliation (prevents state drift)
            self.reconcile_broker_state()

            
            # Agent governance checkpoint
            try:
                if self.agent_master:
                    approved, reason = self.agent_master.evaluate_action({"type": "TRADE_SCAN", "proposed_exposure_pct": 0.10})
                    if not approved:
                        self._emit(f"🧠 Agent blocked scan cycle: {reason}", level="WARN", category="RISK", throttle_key="agent_block_scan", throttle_sec=30)
                        time.sleep(1)
                        continue
            except Exception:
                pass
          
            # Normal Buy Scan
            self.scan_market_parallel()
            # Release C: resolve confirmation-required setups
            self.process_pending_confirmations()

            # Release D1: monitor open entry orders and only log trades on fill
            self.process_pending_orders()
            
            if scan_counter % 5 == 0: 
                self.manage_active_positions_advanced()
            else:
                self.manage_active_positions() 
            
            scan_counter += 1

            # Release C2: periodic snapshot logging (helps diagnose “silent” loops)
            self._emit_snapshot_if_due()
            
            sleep_time = int(self.config['CONFIGURATION'].get('update_interval_sec', 60))
            for _ in range(sleep_time):
                if not self.active: break
                time.sleep(1)
                if _ % 5 == 0: self.check_telegram_commands() 
                if _ % 10 == 0:
                    try:
                        self._e5_watchdog_tick()
                    except Exception:
                        pass
                if _ % int(self._live_order_poll_sec) == 0:
                    self.process_pending_orders()
                if _ % 15 == 0:
                    self.reconcile_broker_state()
                if _ % 30 == 0:
                    self._emit_snapshot_if_due()
                
        # Release E3: one final snapshot + run summary export
        try:
            self._emit_snapshot_if_due(force=True)
        except Exception:
            pass
        try:
            if getattr(self, '_run_summary_enabled', True):
                self.export_run_summary()
        except Exception as e:
            try:
                self._emit(f"⚠️ Run summary export failed: {e}", level='WARN', category='SYSTEM')
            except Exception:
                pass

        self._emit("💤 Engine Stopped.", category="SYSTEM")

    def check_kill_switch(self):
        try:
            acct = self.retry_api_call(self.api.get_account)
            curr_equity = float(acct.equity)
            self.current_equity = curr_equity 
            
            compounding = self.config['CONFIGURATION'].get('compounding_enabled', 'False').lower() == 'true'
            
            if compounding:
                effective_limit = self.start_equity * self.kill_switch_ratio
            else:
                effective_limit = float(self.config['CONFIGURATION'].get('max_daily_loss', 100))

            loss = self.start_equity - curr_equity
            
            if loss > effective_limit:
                self.log(f"☠️ KILL SWITCH! Loss: ${loss:.2f} > Limit: ${effective_limit:.2f}")
                self.send_telegram(f"☠️ KILL SWITCH! Loss: ${loss:.2f}")
                self.liquidate_all()
                self.stop(reason="KILL_SWITCH")
                
        except Exception as e:
            self._emit(
                f"Kill Switch Check Fail: {e}",
                level="ERROR",
                category="RISK",
                throttle_key="kill_switch_check_fail",
                throttle_sec=120,
            )

    def liquidate_all(self):
        # Release D1: cancel open entry orders before liquidating positions
        try:
            self.cancel_all_pending_orders()
        except Exception:
            pass
        trades = self.db.get_active_trades()
        for symbol, data in trades.items():
            self.execute_sell(symbol, data['qty'], 0)
        self._emit("⚠️ All positions liquidated.", level="WARN", category="ORDER")

    # --- Fix 2: gap-aware catch-up insert for missing minute bars ---
    def _fetch_bars_range(self, symbol, is_crypto, start_dt, end_dt):
        """Fetch 1Min bars between start_dt and end_dt (UTC), chunked to avoid API limits."""
        try:
            if start_dt is None or end_dt is None:
                return pd.DataFrame()
            if getattr(start_dt, 'tzinfo', None) is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if getattr(end_dt, 'tzinfo', None) is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if start_dt >= end_dt:
                return pd.DataFrame()

            dfs = []
            chunk_minutes = 9000  # ~6.25 days, stays under common 10k bar limits
            cur = start_dt
            while cur < end_dt:
                chunk_end = min(cur + timedelta(minutes=chunk_minutes), end_dt)
                start_iso = cur.isoformat().replace('+00:00', 'Z')
                end_iso = chunk_end.isoformat().replace('+00:00', 'Z')
                try:
                    if is_crypto:
                        res = self.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', start=start_iso, end=end_iso, limit=10000)
                    else:
                        res = self.retry_api_call(self._api_get_bars_equity, symbol, '1Min', start=start_iso, end=end_iso, limit=10000, adjustment='raw')
                    df = getattr(res, 'df', pd.DataFrame())
                    if df is not None and not df.empty:
                        dfs.append(df)
                except Exception:
                    pass
                # advance; add 1 minute to avoid re-fetching the last bar of the chunk
                cur = chunk_end + timedelta(minutes=1)

            if not dfs:
                return pd.DataFrame()
            df_all = pd.concat(dfs)
            try:
                df_all = df_all[~df_all.index.duplicated(keep='first')]
                df_all = df_all.sort_index()
            except Exception:
                pass
            return df_all
        except Exception:
            return pd.DataFrame()

    def _catch_up_insert_missing_bars(self, symbol, last_db_ts, is_crypto):
        """If gaps exist, fetch missing minute bars, compute indicators, and bulk insert into historical_prices."""
        try:
            if last_db_ts is None:
                return 0

            if getattr(last_db_ts, 'tzinfo', None) is None:
                last_db_ts = last_db_ts.replace(tzinfo=timezone.utc)

            end_dt = datetime.now(timezone.utc)

            # Lookback buffer for indicator stability (EMA200 etc.)
            start_dt = last_db_ts - timedelta(minutes=260)
            if start_dt < end_dt - timedelta(days=30):
                # Safety cap to avoid extremely large runtime catch-ups
                start_dt = end_dt - timedelta(days=30)

            df = self._fetch_bars_range(symbol, is_crypto, start_dt, end_dt)
            if df is None or df.empty or len(df) < 210:
                return 0

            df = df[~df.index.duplicated(keep='first')].sort_index()

            try:
                if getattr(df.index, 'tz', None) is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
            except Exception:
                pass

            bb = ta.bbands(df['close'], length=20, std=2.0)
            rsi = ta.rsi(df['close'], length=14)
            ema = ta.ema(df['close'], length=200)
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            atr_df = ta.atr(df['high'], df['low'], df['close'], length=14)

            if bb is None or rsi is None or ema is None or adx_df is None or atr_df is None:
                return 0

            lower_cols = [c for c in bb.columns if str(c).startswith('BBL')]
            upper_cols = [c for c in bb.columns if str(c).startswith('BBU')]
            if not lower_cols or not upper_cols:
                return 0
            adx_cols = [c for c in adx_df.columns if str(c).startswith('ADX')]
            if not adx_cols:
                return 0

            df['bb_lower'] = bb[lower_cols[0]]
            df['bb_upper'] = bb[upper_cols[0]]
            df['rsi'] = rsi
            df['ema_200'] = ema
            df['adx'] = adx_df[adx_cols[0]]
            df['atr'] = atr_df
            df = df.dropna()

            try:
                last_ts = pd.Timestamp(last_db_ts).tz_convert('UTC') if pd.Timestamp(last_db_ts).tzinfo else pd.Timestamp(last_db_ts).tz_localize('UTC')
            except Exception:
                last_ts = None

            if last_ts is not None:
                df = df[df.index > last_ts]

            if df.empty:
                return 0

            bulk_data = []
            for idx, row in df.iterrows():
                ts = idx.to_pydatetime()
                bulk_data.append((
                    symbol.upper(), ts, float(row['close']), float(row['open']),
                    float(row['high']), float(row['low']), float(row['volume']),
                    float(row['rsi']), float(row['bb_lower']), float(row['bb_upper']),
                    float(row['ema_200']), float(row['adx']), float(row['atr'])
                ))

            if bulk_data:
                inserted = int(self.db.save_bulk_data(symbol, bulk_data) or 0)
                return inserted
            return 0
        except Exception:
            return 0


    def process_symbol_scan(self, symbol, forced_strategy=None):
        try:
            symbol = symbol.upper()
            def _scan_reject(reason: str):
                # Release E2: optional debug trail for early skips (throttled)
                try:
                    if not getattr(self, '_log_decision_rejects', False):
                        return
                    lvl = 'INFO' if getattr(self, '_log_decision_rejects', False) else 'DEBUG'
                    self._emit(
                        f"⏭️ SKIP | {reason}",
                        level=lvl,
                        category='DECISION',
                        symbol=symbol,
                        throttle_key=f"skip_{symbol}_{reason}",
                        throttle_sec=300
                    )
                except Exception:
                    pass
            is_crypto = '/' in symbol
            
            try:
                if is_crypto:
                    bars = self.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=300).df
                else:
                    bars = self.retry_api_call(self._api_get_bars_equity, symbol, '1Min', limit=300).df
            except Exception as e:
                _scan_reject(f"Data fetch failed: {type(e).__name__}")
                return None

            if bars.empty or len(bars) < 210:
                _scan_reject("Insufficient bars (need >=210 for EMA200)")
                return None
            bars = bars[~bars.index.duplicated(keep='first')]

            bb = ta.bbands(bars['close'], length=20, std=2.0)
            rsi = ta.rsi(bars['close'], length=14)
            ema = ta.ema(bars['close'], length=200)
            adx_df = ta.adx(bars['high'], bars['low'], bars['close'], length=14)
            atr_df = ta.atr(bars['high'], bars['low'], bars['close'], length=14)
            
            if bb is None or rsi is None or ema is None or adx_df is None:
                _scan_reject("Indicator computation failed")
                return None
            
            price = bars['close'].iloc[-1]
            ts = bars.index[-1].to_pydatetime()
            try:
                self._record_last_bar_epoch(symbol, ts)
            except Exception:
                pass
            # Prefer explicit BB column names (pandas_ta column order can vary by version)
            try:
                lower_cols = [c for c in bb.columns if str(c).startswith('BBL')]
                upper_cols = [c for c in bb.columns if str(c).startswith('BBU')]
                if not lower_cols or not upper_cols:
                    _scan_reject("BB column missing")
                    return None
                bbl = float(bb[lower_cols[0]].iloc[-1])
                bbu = float(bb[upper_cols[0]].iloc[-1])
            except Exception:
                _scan_reject("BB parse failed")
                return None

            curr_rsi = rsi.iloc[-1]
            curr_ema = ema.iloc[-1]
            adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
            curr_adx = adx_df[adx_col].iloc[-1]
            curr_volume = bars['volume'].iloc[-1]
            curr_atr = atr_df.iloc[-1] if atr_df is not None else 0.0
            
            vol_avg = bars['volume'].rolling(20).mean().iloc[-1]

            # v4.9.0: extra indicators for expanded strategy library
            rsi2_ser = ta.rsi(bars['close'], length=2)
            ema20_ser = ta.ema(bars['close'], length=20)
            ema50_ser = ta.ema(bars['close'], length=50)

            def _safe_float(v, default=0.0):
                try:
                    if v is None:
                        return float(default)
                    if hasattr(v, 'item'):
                        v = v.item()
                    return float(v) if v == v else float(default)
                except Exception:
                    return float(default)

            curr_rsi2 = _safe_float(rsi2_ser.iloc[-1] if rsi2_ser is not None else None, default=50.0)
            curr_ema20 = _safe_float(ema20_ser.iloc[-1] if ema20_ser is not None else None, default=curr_ema)
            curr_ema50 = _safe_float(ema50_ser.iloc[-1] if ema50_ser is not None else None, default=curr_ema)

            # Donchian breakout reference (previous N-bar high)
            try:
                don_len = 20
                don_high_ser = bars['high'].rolling(don_len).max().shift(1)
                curr_don_high = _safe_float(don_high_ser.iloc[-1], default=0.0)
            except Exception:
                curr_don_high = 0.0

            # Simple rolling z-score (for mean reversion)
            try:
                z_len = 20
                mu = bars['close'].rolling(z_len).mean()
                sd = bars['close'].rolling(z_len).std()
                z_ser = (bars['close'] - mu) / sd
                curr_z = _safe_float(z_ser.iloc[-1], default=0.0)
            except Exception:
                curr_z = 0.0

            # Rolling VWAP (windowed, not session-reset)
            try:
                vw = min(390, len(bars))
                pv = (bars['close'] * bars['volume']).rolling(vw).sum()
                vv = bars['volume'].rolling(vw).sum()
                vwap_ser = pv / vv
                curr_vwap = _safe_float(vwap_ser.iloc[-1], default=price)
            except Exception:
                curr_vwap = price

            # Fix 2: catch-up insert if DB has a gap (prevents missing-minute spikes after restarts)
            try:
                last_db_ts = self.db.get_last_timestamp(symbol)
                if last_db_ts is not None:
                    ts_utc = ts
                    if getattr(ts_utc, 'tzinfo', None) is None:
                        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
                    if getattr(last_db_ts, 'tzinfo', None) is None:
                        last_db_ts = last_db_ts.replace(tzinfo=timezone.utc)
                    gap_sec = (ts_utc - last_db_ts).total_seconds()
                    if gap_sec > 70:
                        inserted = self._catch_up_insert_missing_bars(symbol, last_db_ts, is_crypto)
                        if inserted and inserted > 0:
                            self._emit(
                                f"🧩 Catch-up inserted {inserted} missing bars.",
                                level="INFO",
                                category="DATA",
                                symbol=symbol,
                                throttle_key=f"catchup_{symbol}",
                                throttle_sec=300
                            )
            except Exception as e:
                try:
                    self._emit(
                        f"⚠️ Catch-up failed: {self._redact(e)}",
                        level="WARN",
                        category="DATA",
                        symbol=symbol,
                        throttle_key=f"catchup_fail_{symbol}",
                        throttle_sec=300
                    )
                except Exception:
                    pass


            self.db.save_snapshot(symbol, ts, price, bars['open'].iloc[-1], bars['high'].iloc[-1], bars['low'].iloc[-1], curr_volume, curr_rsi, bbl, bbu, curr_ema, curr_adx, curr_atr)
            self.update_chart(symbol) # Ensure UI Updates
            
            strat_name = forced_strategy if forced_strategy else self.optimizer.choose_strategy(
                symbol,
                bars,
                market_regime=getattr(self, 'market_status', 'BULL'),
                is_crypto=is_crypto
            )
            df_history = self.db.get_history(symbol, limit=300) 
            current_bar_data = {
                'close': price, 'open': bars['open'].iloc[-1], 'high': bars['high'].iloc[-1], 'low': bars['low'].iloc[-1],
                'volume': curr_volume,
                'volume_avg': vol_avg,
                'rsi': curr_rsi,
                'rsi2': curr_rsi2,
                'bb_lower': bbl,
                'bb_upper': bbu,
                'ema_20': curr_ema20,
                'ema_50': curr_ema50,
                'ema_200': curr_ema,
                'adx': curr_adx,
                'atr': curr_atr,
                'donchian_high': curr_don_high,
                'zscore': curr_z,
                'vwap': curr_vwap
            }
            
            score = self.optimizer.score_opportunity(symbol, current_bar_data, df_history, strat_name)
            
            sentiment_score = 0.0
            ai_prob = 0.5
            decision = "SKIP"
            reject_reason = "Low Score"

            if score > 0:
                sentiment_score, sentiment_label = self.sentinel.get_sentiment(symbol)
                ai_prob = self.ai.predict_probability(current_bar_data)
                
                if sentiment_score < -0.2:
                    decision = "REJECT"
                    reject_reason = f"Sentiment: {sentiment_label}"
                elif ai_prob < self._cfg_float("ai_min_prob", 0.5):
                    decision = "REJECT"
                    reject_reason = f"AI Veto: {ai_prob:.2f}"
                else:
                    # Release E4: diversification framework (optional gating)
                    if getattr(self, '_e4_enabled', False):
                        ok_cluster, reason_cluster = self._e4_cluster_allows_entry(symbol)
                        if not ok_cluster:
                            decision = "REJECT"
                            reject_reason = reason_cluster
                        else:
                            ok_mtf, reason_mtf = self._e4_mtf_allows_trade(symbol, strat_name, is_crypto)
                            if not ok_mtf:
                                decision = "REJECT"
                                reject_reason = reason_mtf
                            else:
                                decision = "BUY"
                                reject_reason = "Approved"
                    else:
                        decision = "BUY"
                        reject_reason = "Approved"
            


            # v5.12.6 updateA: Decision Packet (why did it trade?)
            decision_id = None
            try:
                decision_id = str(uuid.uuid4())
                thresholds = {
                    'ai_min_prob': self._cfg_float('ai_min_prob', 0.5),
                    'sentiment_reject_below': -0.2,
                    'score_min': 0.0,
                }
                payload = {
                    'bar_timestamp': str(ts),
                    'scan_cycle': int(getattr(self, '_scan_cycle', 0) or 0),
                    'market_regime': str(getattr(self, 'market_status', 'UNKNOWN')),
                    'is_crypto': bool(is_crypto),
                    'strategy': str(strat_name),
                    'score': float(score) if score is not None else 0.0,
                    'decision': str(decision),
                    'reject_reason': str(reject_reason),
                    'ai_prob': float(ai_prob) if ai_prob is not None else 0.0,
                    'sentiment': float(sentiment_score) if sentiment_score is not None else 0.0,
                    'features': dict(current_bar_data) if isinstance(current_bar_data, dict) else {},
                    'thresholds': thresholds,
                    'filters': {
                        'sentiment_ok': (sentiment_score >= thresholds['sentiment_reject_below']) if score > 0 else None,
                        'ai_ok': (ai_prob >= thresholds['ai_min_prob']) if score > 0 else None,
                        'e4_enabled': bool(getattr(self, '_e4_enabled', False)),
                    },
                }
                if hasattr(self.db, 'log_decision_packet'):
                    self.db.log_decision_packet(
                        decision_id=decision_id,
                        symbol=symbol,
                        strategy=strat_name,
                        action=decision,
                        score=score,
                        price=price,
                        ai_prob=ai_prob,
                        sentiment=sentiment_score,
                        reason=reject_reason,
                        market_regime=getattr(self, 'market_status', 'UNKNOWN'),
                        is_crypto=is_crypto,
                        payload=payload,
                    )
            except Exception:
                decision_id = None
            self.db.log_decision(symbol, strat_name, decision, price, curr_rsi, ai_prob, sentiment_score, reject_reason)
            # Release C: decision transparency (BUY always INFO; rejects DEBUG)
            if decision == "BUY":
                self._emit(
                    f"✅ BUY SIGNAL ({strat_name}) | ${price:.2f} | Score {score:.1f} | AI {ai_prob:.2f} | Sent {sentiment_score:.2f}",
                    level="INFO",
                    category="DECISION",
                    symbol=symbol
                )
            else:
                lvl = "INFO" if self._log_decision_rejects else "DEBUG"
                self._emit(
                    f"⏭️ {decision} ({strat_name}) | Score {score:.1f} | AI {ai_prob:.2f} | Sent {sentiment_score:.2f} | {reject_reason}",
                    level=lvl,
                    category="DECISION",
                    symbol=symbol,
                    throttle_key=f"dec_{symbol}_{decision}",
                    throttle_sec=60
                )

            if decision == "BUY":
                return (score, symbol, price, strat_name, curr_atr, ai_prob, decision_id) 
            
            return None
            
        except Exception as e:
            self._emit(
                f"Decision evaluation failed for {symbol}: {type(e).__name__}: {e}",
                level="DEBUG",
                category="DECISION",
                throttle_key=f"decision_eval_fail_{symbol}",
                throttle_sec=120,
            )
            return None

    def scan_market_parallel(self):
        if 'WATCHLIST' not in self.config: return
        
        # v3.8: Shapeshifter Logic
        # Select Watchlist (Phase 4 v5.14.0: ACTIVE universe)
        try:
            from ..watchlist_api import get_watchlist_symbols
            symbols = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL"))
        except Exception:
            symbols = []

        opportunities = []
        scan_started = time.time()
        try:
            self._scan_cycle = int(getattr(self, '_scan_cycle', 0) or 0) + 1
        except Exception:
            self._scan_cycle = 1
        cycle = self._scan_cycle
        n_candidates = 0
        n_confirm_added = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.process_symbol_scan, sym): sym for sym in symbols}
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    score, symbol, price, strat_name, atr, ai_prob, decision_id = res
                    if self._log_candidate_lines:
                        self._emit(
                            f"{symbol} ({strat_name}): ${price:.2f} | Score: {score:.1f} | AI: {ai_prob:.2f}",
                            level="DEBUG",
                            category="CANDIDATE",
                            symbol=symbol,
                            throttle_key=f"cand_{symbol}",
                            throttle_sec=30
                        )
                    n_candidates += 1
                    if score > 0:
                        if self.optimizer.requires_confirmation(strat_name):
                             if symbol not in self.pending_confirmations:
                                 self.pending_confirmations[symbol] = {'time': datetime.now(), 'strategy': strat_name, 'hits': 1, 'last_decision_id': decision_id}
                                 self._emit("🧾 Confirmation required. Waiting for second signal.", level="DEBUG", category="CONFIRM", symbol=symbol, throttle_key=f"conf_add_{symbol}", throttle_sec=60)
                                 n_confirm_added += 1
                        else:
                            opportunities.append(res)
        
        opportunities.sort(key=lambda x: x[0], reverse=True)
        buys = 0
        for score, symbol, price, strat_name, atr, ai_prob, decision_id in opportunities:
            # Release D1: avoid duplicate orders while a prior entry is still open
            if symbol in self._pending_symbols:
                if score > 50:
                    self._emit("⏳ Pending order already open. Skipping.", level="DEBUG", category="ORDER", symbol=symbol, throttle_key=f"pending_{symbol}", throttle_sec=30)
                continue
            allowed, reason = self.wallet.can_buy(symbol, price, self.current_equity)
            if allowed:
                qty = self.wallet.get_trade_qty(price, atr, self.current_equity)
                if qty > 0:
                    self.execute_buy(symbol, qty, price, strat_name, atr, ai_prob, decision_id=decision_id)
                    buys += 1
            elif score > 50:
                 self._emit(f"⚠️ Skipped: {reason}", level="WARN", category="RISK", symbol=symbol, throttle_key=f"skip_{symbol}_{reason}", throttle_sec=60)

        # Release C2: scan summary line + cache stats for snapshot
        try:
            elapsed = time.time() - scan_started
        except Exception:
            elapsed = 0.0
        self._last_scan_stats = {
            'cycle': cycle,
            'symbols': len(symbols),
            'candidates': n_candidates,
            'opps': len(opportunities),
            'buys': buys,
            'confirm_added': n_confirm_added,
            'elapsed_sec': elapsed,
        }

        if self._log_scan_summary:
            self._emit(
                f"🔎 Scan Summary | cycle={cycle} | symbols={len(symbols)} candidates={n_candidates} opps={len(opportunities)} buys={buys} confirm_added={n_confirm_added} | {elapsed:.2f}s",
                # Release E1: make scan summary visible at default INFO level
                level="INFO",
                category="SCAN",
                throttle_key="scan_summary",
                throttle_sec=30
            )

    # --------------------
    # Release C: Confirmation-required strategies (previously unlinked)
    # --------------------
    def process_pending_confirmations(self):
        """Re-check confirmation-required signals and execute buys on a second hit."""
        if not self.pending_confirmations:
            return

        cfg = self.config['CONFIGURATION'] if 'CONFIGURATION' in self.config else {}
        try:
            ttl_minutes = int(float(cfg.get('confirmation_ttl_minutes', 10)))
        except Exception:
            ttl_minutes = 10
        try:
            required_hits = int(float(cfg.get('confirmation_required_hits', 2)))
        except Exception:
            required_hits = 2

        now = datetime.now()
        trades = self.db.get_active_trades()

        for symbol, meta in list(self.pending_confirmations.items()):
            try:
                if symbol in trades:
                    del self.pending_confirmations[symbol]
                    continue

                age_min = (now - meta.get('time', now)).total_seconds() / 60.0
                if age_min > ttl_minutes:
                    del self.pending_confirmations[symbol]
                    self._emit("⏱️ Confirmation expired.", level="DEBUG", category="CONFIRM", symbol=symbol, throttle_key=f"conf_exp_{symbol}", throttle_sec=30)
                    continue


                # Require time separation between first hit and re-check
                try:
                    min_delay_sec = int(float(cfg.get('confirmation_min_delay_seconds', 60)))
                except Exception:
                    min_delay_sec = 60

                try:
                    age_sec = (now - meta.get('time', now)).total_seconds()
                except Exception:
                    age_sec = 999999

                if age_sec < min_delay_sec:
                    self._emit("⏳ Waiting for confirmation window...", level="DEBUG", category="CONFIRM", symbol=symbol, throttle_key=f"conf_wait_{symbol}", throttle_sec=60)
                    continue

                forced_strat = meta.get('strategy')
                res = self.process_symbol_scan(symbol, forced_strategy=forced_strat)
                if not res:
                    self._emit("🔎 Confirmation check: no signal.", level="DEBUG", category="CONFIRM", symbol=symbol, throttle_key=f"conf_none_{symbol}", throttle_sec=60)
                    continue

                score, sym, price, strat_name, atr, ai_prob, decision_id = res
                meta['hits'] = int(meta.get('hits', 1)) + 1
                meta['last_decision_id'] = decision_id

                self._emit(f"✅ Confirmation hit {meta['hits']}/{required_hits} | Score {score:.1f} | AI {ai_prob:.2f}", level="DEBUG", category="CONFIRM", symbol=sym)

                if meta['hits'] >= required_hits:
                    allowed, reason = self.wallet.can_buy(sym, price, self.current_equity)
                    if allowed:
                        if sym in self._pending_symbols:
                            self._emit("⏳ Pending order already open. Skipping.", level="DEBUG", category="CONFIRM", symbol=sym, throttle_key=f"conf_pending_{sym}", throttle_sec=60)
                            del self.pending_confirmations[sym]
                            continue
                        qty = self.wallet.get_trade_qty(price, atr, self.current_equity, ai_score=ai_prob)
                        if qty > 0:
                            self.execute_buy(sym, qty, price, strat_name, atr, ai_prob, decision_id=decision_id)
                    else:
                        self._emit(f"⚠️ Confirmed but denied: {reason}", level="WARN", category="CONFIRM", symbol=sym, throttle_key=f"conf_den_{sym}", throttle_sec=60)

                    del self.pending_confirmations[sym]
            except Exception as e:
                self._emit(
                    f"Confirmation error ({type(e).__name__}): {self._redact(e)}",
                    level="WARN",
                    category="CONFIRM",
                    symbol=symbol,
                    throttle_key=f"conf_err_{symbol}",
                    throttle_sec=60,
                )


    def manage_active_positions_advanced(self):
        trades = self.db.get_active_trades()
        if not trades: return
        if self._log_guardian_banner:
            self._emit("🛡️ Running Guardian Checks (Sentiment & Stagnation)...", level="DEBUG", category="GUARDIAN", throttle_key="guardian_banner", throttle_sec=300)
        
        for symbol, data in trades.items():
            try:
                if '/' in symbol:
                    bar = self.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                    curr_price = bar['close']
                else:
                    bar = self.retry_api_call(self.api.get_latest_bar, symbol)
                    curr_price = bar.c
                
                qty = data['qty']
                entry_price = data['entry_price']
                highest = data['highest_price']
                entry_time_str = data['timestamp'] 
                
                if isinstance(entry_time_str, str):
                    entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    entry_time = entry_time_str

                section = f"STRATEGY_{data['strategy']}"
                if data['strategy'] == "IMPORTED": section = "STRATEGY_THE_GENERAL"
                trail_pct = float(self.config[section]['trailing_stop_pct']) if section in self.config else 0.02
                
                if curr_price > highest:
                    self.db.update_highest_price(symbol, curr_price)
                    highest = curr_price
                stop_price = highest * (1 - trail_pct)
                
                if curr_price < stop_price:
                    self.log(f"📉 Trailing Stop: {symbol} (High: ${highest:.2f}, Stop: ${stop_price:.2f})")
                    self.execute_sell(symbol, qty, curr_price)
                    continue 
                sell_on_news = False
                sent_thresh = -0.3
                try:
                    sell_on_news = self.config['CONFIGURATION'].get('guardian_sell_on_negative_sentiment', 'False').lower() == 'true'
                    sent_thresh = float(self.config['CONFIGURATION'].get('guardian_sentiment_threshold', -0.3))
                except:
                    pass

                if sell_on_news:
                    score, label = self.sentinel.get_sentiment(symbol)
                    if score < sent_thresh:
                        self.log(f"📰 Guardian News Exit: {symbol} (Sentiment: {score:.2f}) - Selling immediately!")
                        self.execute_sell(symbol, qty, curr_price)
                        continue
                duration = (datetime.now() - entry_time).total_seconds() / 60 
                pct_gain = (curr_price - entry_price) / entry_price

                use_stag = True
                stag_minutes = 60
                stag_min = -0.01
                stag_max = 0.003
                try:
                    use_stag = self.config['CONFIGURATION'].get('guardian_use_stagnation_exit', 'True').lower() == 'true'
                    stag_minutes = int(float(self.config['CONFIGURATION'].get('guardian_stagnation_minutes', 60)))
                    stag_min = float(self.config['CONFIGURATION'].get('guardian_stagnation_min_gain', -0.01))
                    stag_max = float(self.config['CONFIGURATION'].get('guardian_stagnation_max_gain', 0.003))
                except:
                    pass

                if use_stag and duration > stag_minutes and stag_min < pct_gain < stag_max:
                    self.log(f"🐌 Stagnation Kill: {symbol} held {int(duration)}m with no move. Freeing capital.")
                    self.execute_sell(symbol, qty, curr_price)
                    continue

            except Exception as e:
                self.log(f"Error managing {symbol}: {e}")

    def manage_active_positions(self):
        trades = self.db.get_active_trades()
        if not trades: return
        for symbol, data in trades.items():
            try:
                if '/' in symbol:
                    bar = self.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                    curr_price = bar['close']
                else:
                    bar = self.retry_api_call(self.api.get_latest_bar, symbol)
                    curr_price = bar.c
                
                highest = data['highest_price']
                qty = data['qty']
                
                strat_name = data['strategy']
                if strat_name == "IMPORTED": strat_name = "THE_GENERAL"
                section = f"STRATEGY_{strat_name}"
                trail_pct = float(self.config[section]['trailing_stop_pct']) if section in self.config else 0.02
                
                if curr_price > highest:
                    self.db.update_highest_price(symbol, curr_price)
                    highest = curr_price
                stop_price = highest * (1 - trail_pct)
                if curr_price < stop_price:
                    self.log(f"📉 Trailing Stop: {symbol} (High: ${highest:.2f}, Stop: ${stop_price:.2f})")
                    self.execute_sell(symbol, qty, curr_price)
            except Exception as e:
                self.log(f"Error managing {symbol}: {e}")

    def execute_buy(self, symbol, qty, price, strat_name, atr=None, ai_prob=0.0, decision_id=None):
        # Release E5: optional safety halt for new entries (exits still allowed)
        if getattr(self, '_entries_disabled', False):
            try:
                self._emit(f"⛔ Entry blocked (halted): {symbol}", level="WARN", category="RISK", symbol=symbol,
                           throttle_key=f"halt_entry_{str(symbol).upper()}", throttle_sec=60)
            except Exception:
                pass
            return
        try:
            # v5.12.6 updateA: execution packet (intent)
            self._log_exec_packet(symbol=symbol, side="BUY", phase="INTENT", decision_id=decision_id, qty=qty, price=price, payload={"strategy": str(strat_name), "atr": atr, "ai_prob": ai_prob, "regime": getattr(self, "market_status", "UNKNOWN")})
            stop_price, take_price = self.optimizer.calculate_exit_prices(price, atr, strat_name)
            is_crypto = '/' in symbol

            # v5.12.5: deterministic idempotency suffix (stable within a bar/minute)
            idem_suffix = self._make_idem_suffix(symbol, strat_name, side='B', price=price)
            client_order_id = self._make_client_order_id(symbol, strat_name, side='B', idem_suffix=idem_suffix)

            # v5.12.5: idempotency adopt - if an open BUY order already exists, track it instead of resubmitting
            existing = self._find_existing_open_entry_order(symbol)
            if existing is not None:
                order = existing
                try:
                    client_order_id = str(getattr(order, 'client_order_id', '') or client_order_id)[:48]
                except Exception:
                    pass
                self._log_exec_packet(symbol=symbol, side="BUY", phase="ADOPT", decision_id=decision_id, qty=qty, price=price, order_id=str(getattr(order, "id", "")), client_order_id=client_order_id, broker_status=str(getattr(order, "status", "")), payload={"reason": "existing_open_order"})
                msg = f"🧾 BUY ADOPTED {qty} {symbol} @ ${price:.2f} [AI: {ai_prob:.2f}]"
            else:
                if stop_price and take_price and not is_crypto:
                    submit_kwargs = dict(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='limit',
                        limit_price=price,
                        time_in_force='gtc',
                        client_order_id=client_order_id,
                        order_class='bracket',
                        stop_loss={'stop_price': stop_price},
                        take_profit={'limit_price': take_price},
                    )
                    order = self._submit_order_idempotent(client_order_id=client_order_id, submit_kwargs=submit_kwargs)
                    self._log_exec_packet(symbol=symbol, side="BUY", phase="SUBMIT", decision_id=decision_id, qty=qty, price=price, order_id=str(getattr(order, 'id', '')), client_order_id=client_order_id, broker_status=str(getattr(order, 'status', '')), payload={"order_class": "bracket", "stop_price": stop_price, "take_price": take_price})
                    msg = f"🛒 BUY SUBMITTED {qty} {symbol} @ ${price:.2f} [AI: {ai_prob:.2f}]"
                else:
                    submit_kwargs = dict(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='limit',
                        limit_price=price,
                        time_in_force='gtc',
                        client_order_id=client_order_id,
                    )
                    order = self._submit_order_idempotent(client_order_id=client_order_id, submit_kwargs=submit_kwargs)
                    self._log_exec_packet(symbol=symbol, side="BUY", phase="SUBMIT", decision_id=decision_id, qty=qty, price=price, order_id=str(getattr(order, 'id', '')), client_order_id=client_order_id, broker_status=str(getattr(order, 'status', '')), payload={"order_class": "bracket", "stop_price": stop_price, "take_price": take_price})
                    msg = f"🛒 BUY SUBMITTED {qty} {symbol} @ ${price:.2f} [AI: {ai_prob:.2f}]"
            # Release D1: do NOT write active_trades until fill (prevents ghost trades)
            try:
                oid = str(getattr(order, 'id', ''))
            except Exception:
                oid = ""
            if oid:
                self.pending_orders[oid] = {
                    'symbol': symbol.upper(),
                    'qty': float(qty),
                    'price': float(price),
                    'strategy': str(strat_name),
                    'submitted_ts': time.time(),
                    'order_class': getattr(order, 'order_class', None),
                    'client_order_id': client_order_id,
                    'decision_id': decision_id,
                }
                self._pending_symbols.add(symbol.upper())

                try:
                    self._order_stats['submitted'] = int(self._order_stats.get('submitted', 0)) + 1
                except Exception:
                    pass

            if self._log_order_lifecycle and oid:
                self._emit(f"{msg} | id={oid}", category="ORDER", symbol=symbol)
            else:
                self._emit(msg, category="ORDER", symbol=symbol)
            self.send_telegram(msg)
        except Exception as e:
            try:
                self._order_stats['rejected'] = int(self._order_stats.get('rejected', 0)) + 1
            except Exception:
                pass
            try:
                self._e5_note_reject(symbol)
            except Exception:
                pass
            try:
                self._log_exec_packet(symbol=symbol, side="BUY", phase="ERROR", decision_id=decision_id, qty=qty, price=price, payload={"error": type(e).__name__})
            except Exception:
                pass
            self._emit(
                f"❌ Buy Failed ({type(e).__name__}) | symbol={symbol} qty={qty} price={price}: {self._redact(e)}",
                level="ERROR",
                category="ORDER",
                symbol=symbol,
            )

    def execute_sell(self, symbol, qty, price):
        try:
            self._log_exec_packet(symbol=symbol, side="SELL", phase="INTENT", decision_id=None, qty=qty, price=price, payload={"note": "close_position"})
            self._close_position_idempotent(symbol)
            time.sleep(1) 
            
            if '/' in symbol:
                 trade_price = price 
            else:
                 trade = self.retry_api_call(self.api.get_latest_trade, symbol)
                 trade_price = trade.price
            
            pl = self.db.close_trade(symbol, trade_price)
            try:
                self._log_exec_packet(symbol=symbol, side="SELL", phase="CLOSE", decision_id=None, qty=qty, price=trade_price, payload={"profit_loss": pl})
            except Exception:
                pass
            msg = f"💰 SOLD {symbol}: ${trade_price:.2f} | P/L: ${pl:.2f}"
            self._emit(msg, category="ORDER", symbol=symbol)
            self.send_telegram(msg)
        except Exception as e:
            try:
                self._log_exec_packet(symbol=symbol, side="SELL", phase="ERROR", decision_id=None, qty=qty, price=price, payload={"error": type(e).__name__})
            except Exception:
                pass
            self._emit(
                f"❌ Sell Failed ({type(e).__name__}) | symbol={symbol} qty={qty} price={price}: {self._redact(e)}",
                level="ERROR",
                category="ORDER",
                symbol=symbol,
            )
