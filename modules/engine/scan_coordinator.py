"""Scan coordinator service extracted from TradingEngine.

Forward-only refactor: this module centralizes scan loop parallelism and
throttled scan-summary logging.

This class intentionally proxies attribute access to the TradingEngine instance
to avoid behavior changes while allowing core.py to shrink.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class ScanCoordinatorService:
    """Thin service wrapper around TradingEngine scan orchestration."""

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

    def scan_market_parallel(self):
        symbols = self._resolve_scan_symbols()

        if not symbols:
            self._emit(
                "🔎 Scan skipped: no symbols available from watchlist/candidates.",
                level="WARN",
                category="SCAN",
                throttle_key="scan_no_symbols",
                throttle_sec=60,
            )
            return

        # v3.8: Shapeshifter Logic
        # Select Watchlist (Phase 4 v5.14.0: ACTIVE universe)

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
                                self.pending_confirmations[symbol] = {
                                    'time': datetime.now(),
                                    'strategy': strat_name,
                                    'hits': 1,
                                    'last_decision_id': decision_id,
                                }
                                self._emit(
                                    "🧾 Confirmation required. Waiting for second signal.",
                                    level="DEBUG",
                                    category="CONFIRM",
                                    symbol=symbol,
                                    throttle_key=f"conf_add_{symbol}",
                                    throttle_sec=60
                                )
                                n_confirm_added += 1
                        else:
                            opportunities.append(res)

        opportunities.sort(key=lambda x: x[0], reverse=True)
        buys = 0
        for score, symbol, price, strat_name, atr, ai_prob, decision_id in opportunities:
            # Release D1: avoid duplicate orders while a prior entry is still open
            if symbol in self._pending_symbols:
                if score > 50:
                    self._emit(
                        "⏳ Pending order already open. Skipping.",
                        level="DEBUG",
                        category="ORDER",
                        symbol=symbol,
                        throttle_key=f"pending_{symbol}",
                        throttle_sec=30
                    )
                continue
            allowed, reason = self.wallet.can_buy(symbol, price, self.current_equity)
            if allowed:
                qty = self.wallet.get_trade_qty(price, atr, self.current_equity)
                if qty > 0:
                    self.execute_buy(symbol, qty, price, strat_name, atr, ai_prob, decision_id=decision_id)
                    buys += 1
            elif score > 50:
                self._emit(
                    f"⚠️ Skipped: {reason}",
                    level="WARN",
                    category="RISK",
                    symbol=symbol,
                    throttle_key=f"skip_{symbol}_{reason}",
                    throttle_sec=60
                )

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

    def _resolve_scan_symbols(self):
        """Resolve scan symbols without relying on legacy [WATCHLIST] config sections.

        Resolution order:
          1) ACTIVE watchlist symbols (primary source of truth).
          2) ALL watchlist symbols (fallback when ACTIVE is empty).
          3) Latest DB candidates (bootstrap fallback).
        """

        def _uniq(items):
            out, seen = [], set()
            for raw in items or []:
                try:
                    sym = str(raw or "").strip().upper()
                except Exception:
                    continue
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                out.append(sym)
            return out

        try:
            from ..watchlist_api import get_watchlist_symbols
            active = _uniq(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL"))
        except Exception:
            active = []

        if active:
            return active

        try:
            from ..watchlist_api import get_watchlist_symbols
            all_symbols = _uniq(get_watchlist_symbols(self.config, group="ALL", asset="ALL"))
        except Exception:
            all_symbols = []

        if all_symbols:
            return all_symbols

        db = getattr(self, "db", None)
        if db is None or not hasattr(db, "get_latest_candidates"):
            return []

        try:
            rows = db.get_latest_candidates(limit=200) or []
            return _uniq((r or {}).get("symbol") for r in rows)
        except Exception:
            return []
