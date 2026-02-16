"""Trading engine core.

This module contains the full TradingEngine implementation.
It was moved here during the v5.x refactor series to reduce blast radius.

"""

from .broker_gateway import BrokerGateway
from .order_lifecycle import OrderLifecycleService
from .risk_manager import RiskManagerService
from .scan_coordinator import ScanCoordinatorService
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
import logging
from datetime import datetime, timedelta, timezone
from collections import deque
from ..strategies import StrategyOptimizer, WalletManager
from ..sentiment import NewsSentinel 
from ..ai import AI_Oracle 
from ..utils import APP_VERSION, APP_RELEASE, get_paths
from ..logging_utils import get_logger

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
        self._py_logger = get_logger(__name__)
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
        self.broker = BrokerGateway(self.config, self._py_logger, note_api_error=self._e5_note_api_error)

        self._connect_broker()
        if self.api:
            self.sentinel = NewsSentinel(self.api)
            
        # v3.9.16 FIX: Pass log_callback so AI speaks to UI
        self.ai = AI_Oracle(self.db, self.config, self.log)
        threading.Thread(target=self.ai_training_thread, daemon=True).start()

    def _current_mode(self) -> str:
        """Best-effort agent mode for structured log context."""
        try:
            am = getattr(self, "agent_master", None)
            if am is not None:
                m = getattr(am, "mode", None)
                if m:
                    return str(m).strip().upper()
        except Exception:
            pass
        try:
            return str(self.config["CONFIGURATION"].get("agent_mode", "OFF")).strip().upper() or "OFF"
        except Exception:
            return "OFF"

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
        return self._get_risk_manager()._reload_e3_config()

    def _reload_e4_config(self):
        return self._get_risk_manager()._reload_e4_config()

    def _reload_e5_config(self):
        return self._get_risk_manager()._reload_e5_config()

    def _e5_is_enforced(self):
        return self._get_risk_manager()._e5_is_enforced()

    def _e5__prune_events(self, dq, now_ts):
        return self._get_risk_manager()._e5__prune_events(dq, now_ts)

    def _e5_note_api_error(self, _kind=None):
        return self._get_risk_manager()._e5_note_api_error(_kind)

    def _e5_note_reject(self, _symbol=None):
        return self._get_risk_manager()._e5_note_reject(_symbol)

    def _e5_note_ttl_cancel(self, _symbol=None):
        return self._get_risk_manager()._e5_note_ttl_cancel(_symbol)

    def _e5_watchdog_tick(self):
        return self._get_risk_manager()._e5_watchdog_tick()

    def _e4_parse_clusters(self, raw: str):
        return self._get_risk_manager()._e4_parse_clusters(raw)

    def _e4_strategy_playbook(self, strat_name: str) -> str:
        return self._get_risk_manager()._e4_strategy_playbook(strat_name)

    def _e4_count_cluster_exposure(self, cluster_name: str) -> int:
        return self._get_risk_manager()._e4_count_cluster_exposure(cluster_name)

    def _e4_cluster_allows_entry(self, symbol: str):
        return self._get_risk_manager()._e4_cluster_allows_entry(symbol)

    def _e4_get_mtf_bars(self, symbol: str, timeframe: str, is_crypto: bool):
        return self._get_risk_manager()._e4_get_mtf_bars(symbol, timeframe, is_crypto)

    def _e4_get_mtf_context(self, symbol: str, is_crypto: bool):
        return self._get_risk_manager()._e4_get_mtf_context(symbol, is_crypto)

    def _e4_mtf_allows_trade(self, symbol: str, strat_name: str, is_crypto: bool):
        return self._get_risk_manager()._e4_mtf_allows_trade(symbol, strat_name, is_crypto)

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

    def _emit(
        self,
        msg,
        level="INFO",
        category=None,
        symbol=None,
        order_id=None,
        strategy=None,
        throttle_key=None,
        throttle_sec=0,
    ):
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
            self.log(
                msg,
                level=level,
                category=category,
                symbol=symbol,
                order_id=order_id,
                strategy=strategy,
                component="engine",
                mode=self._current_mode(),
            )
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
            if order_id:
                parts.append(f"OID:{order_id}")
            if strategy:
                parts.append(f"STRAT:{strategy}")

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
        return self._get_risk_manager()._emit_snapshot_if_due(force=force)

    def _record_last_bar_epoch(self, symbol, ts):
        return self._get_risk_manager()._record_last_bar_epoch(symbol, ts)

    def _connect_broker(self):
        """Connect to Alpaca via BrokerGateway.

        BrokerGateway owns connection + retry mechanics; engine core remains orchestration.
        """
        # Reset API surface
        self.api = None

        try:
            # Ensure the gateway exists (defensive for tests / stubs)
            if not hasattr(self, 'broker') or self.broker is None:
                self.broker = BrokerGateway(self.config, self._py_logger, note_api_error=self._e5_note_api_error)

            # Connect gateway and expose it as the engine's API surface.
            self.broker.connect()
            self.api = self.broker

            acct = self.api.retry_api_call(self.api.get_account)
            if acct:
                self.start_equity = float(getattr(acct, 'last_equity', 0.0))
                self.current_equity = float(getattr(acct, 'equity', 0.0))
                base_url = getattr(self.broker, 'base_url', '') or ''
                self._emit(f"✅ API Connected ({base_url})", category="SYSTEM")
            else:
                self._emit("⚠️ API Connected but account fetch failed", category="SYSTEM")

            self.sync_positions()

        except ValueError as e:
            self.api = None
            self._emit("ℹ️ API not configured (set Alpaca keys in config/keys.ini)", category="SYSTEM")
            self._py_logger.warning("[W_ENGINE_API_NOT_CONFIGURED] %s", e)
            return

        except FileNotFoundError as e:
            self.api = None
            self._emit("ℹ️ keys.ini not found (broker disabled)", category="SYSTEM")
            self._py_logger.warning("[W_ENGINE_KEYS_MISSING] %s", e)
            return

        except Exception:
            self.api = None
            self._py_logger.exception("[E_ENGINE_API_CONNECT_FAIL] Alpaca API connect failed")
            self._emit("❌ API connect failed (check keys.ini)", category="SYSTEM")
            return


    def sync_positions(self):
        try:
            self._emit("🔄 Syncing Alpaca positions to Database...", category="SYNC")
            alpaca_positions = self.api.retry_api_call(self.api.list_positions)
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

    def _get_order_lifecycle(self):
        """Lazy initializer for OrderLifecycleService (supports test stubs that bypass __init__)."""
        svc = getattr(self, 'order_lifecycle', None)
        if svc is None:
            svc = OrderLifecycleService(self)
            self.order_lifecycle = svc
        return svc



    def _get_risk_manager(self):
        """Lazy initializer for RiskManagerService (supports test stubs that bypass __init__)."""
        svc = getattr(self, 'risk_manager', None)
        if svc is None:
            svc = RiskManagerService(self)
            self.risk_manager = svc
        return svc

    def _get_scan_coordinator(self):
        """Lazy initializer for ScanCoordinatorService (supports test stubs that bypass __init__)."""
        svc = getattr(self, 'scan_coordinator', None)
        if svc is None:
            svc = ScanCoordinatorService(self)
            self.scan_coordinator = svc
        return svc

    def _get_order_by_client_order_id_safe(self, client_order_id):
        return self._get_order_lifecycle()._get_order_by_client_order_id_safe(client_order_id)

    def _adopt_open_buy_order(self, o):
        return self._get_order_lifecycle()._adopt_open_buy_order(o)

    def _find_existing_open_entry_order(self, symbol):
        return self._get_order_lifecycle()._find_existing_open_entry_order(symbol)

    def _submit_order_idempotent(self, *, client_order_id: str, submit_kwargs: dict):
        return self._get_order_lifecycle()._submit_order_idempotent(client_order_id=client_order_id, submit_kwargs=submit_kwargs)

    def _close_position_idempotent(self, symbol: str) -> None:
        return self._get_order_lifecycle()._close_position_idempotent(symbol)

    def reconcile_broker_state(self, force: bool = False):
        return self._get_order_lifecycle().reconcile_broker_state(force=force)

    def sync_open_orders(self):
        return self._get_order_lifecycle().sync_open_orders()

    def process_pending_orders(self):
        return self._get_order_lifecycle().process_pending_orders()

    def cancel_all_pending_orders(self):
        return self._get_order_lifecycle().cancel_all_pending_orders()

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
                    bar = self.api.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                else:
                    bar = self.api.retry_api_call(self.api.get_latest_bar, symbol)
                
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
                base_url = (self.config.get('KEYS', {}).get('base_url', '') or '').lower()
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
        
        self._connect_broker()
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
        return self._get_risk_manager().check_kill_switch()

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
                        res = self.api.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', start=start_iso, end=end_iso, limit=10000)
                    else:
                        res = self.api.retry_api_call(self._api_get_bars_equity, symbol, '1Min', start=start_iso, end=end_iso, limit=10000, adjustment='raw')
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
                    bars = self.api.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=300).df
                else:
                    bars = self.api.retry_api_call(self._api_get_bars_equity, symbol, '1Min', limit=300).df
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
        return self._get_scan_coordinator().scan_market_parallel()

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
                    bar = self.api.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                    curr_price = bar['close']
                else:
                    bar = self.api.retry_api_call(self.api.get_latest_bar, symbol)
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
                    bar = self.api.retry_api_call(self.api.get_crypto_bars, symbol, '1Min', limit=1).df.iloc[-1]
                    curr_price = bar['close']
                else:
                    bar = self.api.retry_api_call(self.api.get_latest_bar, symbol)
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
                 trade = self.api.retry_api_call(self.api.get_latest_trade, symbol)
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
