
"""Order lifecycle service extracted from TradingEngine.

Forward-only refactor: this module centralizes order submission/close idempotency,
TTL cancels, and broker reconciliation logic.

This class intentionally proxies attribute access to the TradingEngine instance
to avoid behavior changes while allowing core.py to shrink.
"""

from __future__ import annotations

import time
import requests
import pandas as pd


class OrderLifecycleService:
    """Thin service wrapper around TradingEngine order lifecycle logic."""

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

    def _get_order_by_client_order_id_safe(self, client_order_id):
        """Best-effort lookup for an order by client_order_id (Alpaca support varies by SDK version)."""
        if not self.api or not client_order_id:
            return None
        try:
            if hasattr(self.api, 'get_order_by_client_order_id'):
                return self.api.retry_api_call(self.api.get_order_by_client_order_id, client_order_id)
        except Exception:
            pass
        # Fallback: scan recent orders
        try:
            orders = self.api.retry_api_call(self.api.list_orders, status='all', limit=500)
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
            orders = self.api.retry_api_call(self.api.list_orders, status='open', limit=500)
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
            positions = self.api.retry_api_call(self.api.list_positions) or []
        except Exception as e:
            self._emit(f"[E_RECON_API_POS] reconcile positions failed: {self._redact(e)}", level="WARN", category="SYNC", throttle_key="recon_pos_fail", throttle_sec=60)
            return
        try:
            open_orders = self.api.retry_api_call(self.api.list_orders, status='open', limit=500) or []
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
                    self._emit(f"[E_RECON_PENDING_MISSING] Dropped missing pending order {oid} for {sym}", level="WARN", category="ORDER", symbol=sym, order_id=oid, throttle_key=f"recon_pend_{sym}", throttle_sec=60)
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
            orders = self.api.retry_api_call(self.api.list_orders, status='open', limit=500)
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
                    pos = self.api.retry_api_call(self.api.get_position, sym)
                    if pos is not None:
                        filled_price = float(getattr(pos, 'avg_entry_price', meta.get('price', 0)) or meta.get('price', 0))
                        qty = float(getattr(pos, 'qty', meta.get('qty', 0)) or meta.get('qty', 0))
                        self.db.log_trade_entry(sym, qty, filled_price, strat)
                        self._log_exec_packet(symbol=sym, side="BUY", phase="FILL", decision_id=meta.get("decision_id"), qty=qty, price=filled_price, order_id=oid, client_order_id=meta.get("client_order_id"), broker_status="filled_via_position", payload={"strategy": strat})
                        try:
                            self._order_stats['filled'] = int(self._order_stats.get('filled', 0)) + 1
                        except Exception:
                            pass
                        self._emit(f"✅ FILLED {sym} | Qty {qty} @ ${filled_price:.2f} | {strat}", level="INFO", category="ORDER", symbol=sym, order_id=oid, strategy=strat)
                        del self.pending_orders[oid]
                        self._pending_symbols.discard(sym)
                        continue
                except Exception:
                    pass

                # Otherwise poll the order itself
                try:
                    o = self.api.retry_api_call(self.api.get_order, oid)
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
                    self._emit(f"✅ FILLED {sym} | Qty {fq} @ ${fp:.2f} | {strat}", level="INFO", category="ORDER", symbol=sym, order_id=oid, strategy=strat)
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
                        self._emit(f"❌ BUY {sym} {status.upper()} | {strat}", level="WARN", category="ORDER", symbol=sym, order_id=oid, strategy=strat)
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
                        self.api.retry_api_call(self.api.cancel_order, oid)
                    except Exception:
                        pass
                    try:
                        self._log_exec_packet(symbol=sym, side="BUY", phase="TTL_CANCEL", decision_id=meta.get("decision_id"), qty=meta.get("qty"), price=meta.get("price"), order_id=oid, client_order_id=meta.get("client_order_id"), broker_status="canceled", payload={"age_sec": age, "strategy": strat})
                    except Exception:
                        pass
                    if self._log_order_lifecycle:
                        self._emit(f"⏱️ CANCELED unfilled BUY {sym} after {int(age)}s | {strat}", level="WARN", category="ORDER", symbol=sym, order_id=oid, strategy=strat, throttle_key=f"ttl_{sym}", throttle_sec=30)
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
                self.api.retry_api_call(self.api.cancel_order, oid)
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

