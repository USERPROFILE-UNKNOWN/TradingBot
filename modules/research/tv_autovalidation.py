"""
modules/research/tv_autovalidation.py

TradingView alert autovalidation (PAPER-only):
  - ensure symbol has fresh market data in historical DB
  - run a quick sanity backtest over last N days
  - persist a compact JSON bundle and index the run in metrics.db

Design goals:
  - non-blocking (runs in background worker threads)
  - robust when keys/creds are missing (skip, never crash)
  - EXE-friendly (no extra deps beyond what project already ships)
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from modules.paths import get_paths
from modules.research.quick_backtest import run_quick_backtest


@dataclass
class _WorkItem:
    enqueued_ts: float
    symbol: str
    payload: Dict[str, Any]
    key: str  # symbol|timeframe|signal


class TradingViewAutoValidator:
    def __init__(
        self,
        config,
        db_manager,
        metrics_store,
        log: Callable[[str], None],
        agent_mode_getter: Callable[[], str],
    ) -> None:
        self.config = config
        self.db = db_manager
        self.metrics = metrics_store
        self.log = log
        self._agent_mode_getter = agent_mode_getter

        self._q: Deque[_WorkItem] = deque()
        self._lock = threading.Lock()
        self._inflight: Dict[str, float] = {}  # key -> start_ts
        self._last_key_ts: Dict[str, float] = {}  # cooldown

    # -----------------------------
    # config helpers
    # -----------------------------
    def _get(self, section: str, key: str, fallback: Any = None) -> Any:
        try:
            if self.config.has_option(section, key):
                return self.config.get(section, key)
        except Exception:
            return fallback
        return fallback

    def _get_bool(self, section: str, key: str, fallback: bool) -> bool:
        v = self._get(section, key, None)
        if v is None:
            return fallback
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return fallback

    def _get_int(self, section: str, key: str, fallback: int) -> int:
        v = self._get(section, key, None)
        if v is None:
            return int(fallback)
        try:
            return int(float(v))
        except Exception:
            return int(fallback)

    def _enabled(self) -> bool:
        # PAPER-only (both global agent mode and TRADINGVIEW.mode)
        try:
            tv_mode = str(self._get("TRADINGVIEW", "mode", "OFF")).strip().upper()
        except Exception:
            tv_mode = "OFF"

        agent_mode = str(self._agent_mode_getter() or "").strip().upper()

        if agent_mode != "PAPER":
            return False
        if tv_mode != "PAPER":
            return False

        # explicit enable flag (optional)
        return self._get_bool("TRADINGVIEW", "autovalidation_enabled", True)

    # -----------------------------
    # public API
    # -----------------------------
    def maybe_enqueue(self, payload: Dict[str, Any]) -> None:
        if not self._enabled():
            return

        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            return

        timeframe = str(payload.get("timeframe") or payload.get("tf") or "").strip().upper()
        signal = str(payload.get("signal") or payload.get("action") or "").strip().upper()
        key = f"{symbol}|{timeframe}|{signal}"

        cooldown_min = self._get_int("TRADINGVIEW", "autovalidation_cooldown_minutes", 10)
        cooldown_sec = max(0, cooldown_min) * 60

        now = time.time()

        with self._lock:
            # cooldown gate
            last = self._last_key_ts.get(key, 0.0)
            if cooldown_sec > 0 and (now - last) < cooldown_sec:
                return

            # don't enqueue if already queued/inflight
            if key in self._inflight:
                return
            if any(item.key == key for item in self._q):
                return

            self._last_key_ts[key] = now
            self._q.append(_WorkItem(enqueued_ts=now, symbol=symbol, payload=dict(payload), key=key))

    def pump(self) -> None:
        """
        Fast scheduler hook: pops queue and starts worker threads.
        Must return quickly (no blocking network/IO).
        """
        if not self._enabled():
            return

        max_conc = self._get_int("TRADINGVIEW", "autovalidation_max_concurrency", 1)
        max_conc = max(1, min(max_conc, 4))

        with self._lock:
            if len(self._inflight) >= max_conc:
                return
            if not self._q:
                return
            item = self._q.popleft()
            self._inflight[item.key] = time.time()

        t = threading.Thread(target=self._worker, args=(item,), daemon=True)
        t.start()

    # -----------------------------
    # worker
    # -----------------------------
    def _worker(self, item: _WorkItem) -> None:
        started = time.time()
        status = "ok"
        err = ""
        bundle_path = ""
        best_strategy = ""
        best_pl = None

        try:
            self.log(f"[TV][AUTO] starting autovalidation: {item.key}")

            # 1) backfill / refresh data
            self._ensure_fresh_data(item.symbol)

            # 2) quick backtest
            bt_days = self._get_int("TRADINGVIEW", "autovalidation_backtest_days", 14)
            max_strats = self._get_int("TRADINGVIEW", "autovalidation_max_strategies", 6)
            min_trades = self._get_int("TRADINGVIEW", "autovalidation_min_trades", 1)

            bundle = run_quick_backtest(
                self.config,
                self.db,
                item.symbol,
                days=bt_days,
                max_strategies=max_strats,
                min_trades=min_trades,
            )

            # 3) persist bundle json
            bundle_path = self._save_bundle(item, bundle)

            best = bundle.get("best") if isinstance(bundle, dict) else None
            if isinstance(best, dict):
                best_strategy = str(best.get("strategy") or "")
                try:
                    best_pl = float(best.get("total_pl"))
                except Exception:
                    best_pl = None

            # 4) index in metrics.db
            try:
                self.metrics.log_tradingview_backtest(
                    symbol=item.symbol,
                    timeframe=str(item.payload.get("timeframe") or ""),
                    signal=str(item.payload.get("signal") or ""),
                    bundle_path=bundle_path,
                    best_strategy=best_strategy,
                    best_pl=best_pl,
                    status="ok" if bundle.get("ok", False) else "error",
                    error=str(bundle.get("error") or ""),
                )
            except Exception as e:
                self.log(f"[TV][AUTO] metrics index failed: {e}")

            self.log(f"[TV][AUTO] done: {item.key} -> {best_strategy or 'N/A'} (bundle saved)")

        except Exception as e:
            status = "error"
            err = str(e)
            self.log(f"[TV][AUTO] failed: {item.key} :: {e}")

            try:
                self.metrics.log_tradingview_backtest(
                    symbol=item.symbol,
                    timeframe=str(item.payload.get("timeframe") or ""),
                    signal=str(item.payload.get("signal") or ""),
                    bundle_path=bundle_path,
                    best_strategy=best_strategy,
                    best_pl=best_pl,
                    status=status,
                    error=err,
                )
            except Exception:
                pass

        finally:
            dur_ms = int((time.time() - started) * 1000)
            if status != "ok":
                self.log(f"[TV][AUTO] completed with errors in {dur_ms}ms: {item.key}")
            else:
                self.log(f"[TV][AUTO] completed in {dur_ms}ms: {item.key}")

            with self._lock:
                self._inflight.pop(item.key, None)

    # -----------------------------
    # internals
    # -----------------------------
    def _ensure_fresh_data(self, symbol: str) -> None:
        freshness_min = self._get_int("TRADINGVIEW", "autovalidation_freshness_minutes", 30)
        freshness_sec = max(1, freshness_min) * 60

        last_ts = None
        try:
            last_ts = self.db.get_last_timestamp(symbol)
        except Exception:
            last_ts = None

        if last_ts is not None:
            try:
                age = (datetime.now(timezone.utc) - last_ts).total_seconds()
                if age <= freshness_sec:
                    return
            except Exception:
                # fallthrough to update attempt
                pass

        backfill_days = self._get_int("TRADINGVIEW", "autovalidation_backfill_days", 60)
        backfill_days = max(1, min(backfill_days, 365))

        api = self._get_alpaca_api()
        if api is None:
            self.log(f"[TV][AUTO] skip backfill for {symbol}: missing Alpaca creds")
            return

        try:
            from modules.market_data.updater import IncrementalUpdater
        except Exception as e:
            self.log(f"[TV][AUTO] updater import failed: {e}")
            return

        try:
            updater = IncrementalUpdater(api=api, config=self.config, db_manager=self.db, log_callback=self.log)
            updater.update_symbol(symbol, days=backfill_days, timeframe="1Min")
        except Exception as e:
            self.log(f"[TV][AUTO] backfill failed for {symbol}: {e}")

    def _get_alpaca_api(self):
        """
        Create Alpaca REST client using config values.

        This must be robust if keys.ini is absent (tests / offline).
        """
        try:
            key = self._get("KEYS", "api_key", "") or self._get("KEYS", "APCA_API_KEY_ID", "")
            secret = self._get("KEYS", "api_secret", "") or self._get("KEYS", "APCA_API_SECRET_KEY", "")
            base_url = self._get("KEYS", "base_url", "") or self._get("KEYS", "APCA_API_BASE_URL", "")
        except Exception:
            key = secret = base_url = ""

        if not key or not secret:
            return None

        try:
            # Lazy import to keep boot path safe if package is missing.
            import alpaca_trade_api as tradeapi  # type: ignore
            if not base_url:
                base_url = "https://paper-api.alpaca.markets"
            return tradeapi.REST(key, secret, base_url, api_version="v2")
        except Exception as e:
            self.log(f"[TV][AUTO] Alpaca client init failed: {e}")
            return None

    def _save_bundle(self, item: _WorkItem, bundle: Dict[str, Any]) -> str:
        paths = get_paths()
        logs_dir = paths.get("logs") or "logs"
        out_dir = f"{logs_dir}\\backtest\\tv_autovalidation"
        try:
            import os
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            # fallback to logs root
            out_dir = logs_dir

        ts = datetime.now(timezone.utc).strftime("%Y.%m.%d_%H.%M.%S")
        safe_symbol = "".join(ch for ch in item.symbol if ch.isalnum() or ch in ("_", "-"))
        fn = f"tv_autobacktest_{safe_symbol}_{ts}.json"
        full = f"{out_dir}\\{fn}"

        payload = {
            "meta": {
                "ts_utc": ts,
                "symbol": item.symbol,
                "key": item.key,
                "source": "tradingview",
                "agent_mode": str(self._agent_mode_getter() or ""),
            },
            "alert": item.payload,
            "bundle": bundle,
        }

        try:
            with open(full, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=False)
        except Exception as e:
            self.log(f"[TV][AUTO] failed to save bundle: {e}")
            return ""

        return full
