
"""Risk management service extracted from TradingEngine.

Forward-only refactor: this module centralizes safety gates and risk controls,
including E3/E4/E5 governance, exposure checks, and kill-switch triggers.

This class intentionally proxies attribute access to the TradingEngine instance
to avoid behavior changes while allowing core.py to shrink.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import pandas_ta as ta


class RiskManagerService:
    """Thin service wrapper around TradingEngine risk / safety logic."""

    def __init__(self, engine):
        object.__setattr__(self, "_engine", engine)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def __setattr__(self, name, value):
        # Forward all state mutations to the engine so behavior remains identical.
        if name == "_engine":
            object.__setattr__(self, name, value)
        else:
            setattr(self._engine, name, value)

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
                    try:
                        self._py_logger.exception(
                            "[E_ENGINE_CANCEL_PENDING_FAIL] cancel_all_pending_orders failed",
                            extra={"component": "engine", "mode": self._current_mode()},
                        )
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
                df = self.api.retry_api_call(self.api.get_crypto_bars, symbol, timeframe, limit=300).df
            else:
                df = self.api.retry_api_call(self._api_get_bars_equity, symbol, timeframe, limit=300).df

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

    def check_kill_switch(self):
        try:
            acct = self.api.retry_api_call(self.api.get_account)
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

