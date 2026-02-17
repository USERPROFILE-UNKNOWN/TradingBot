"""AI Master Agent orchestration with guarded paper/live controls."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .event_bus import EventBus
from .scheduler import JobScheduler
from .governance import Governance
from .metrics import MetricsStore
from .logging_utils import get_component_logger
from .research.tv_autovalidation import TradingViewAutoValidator


class AgentMaster:
    MODES = ("OFF", "ADVISORY", "PAPER", "LIVE")

    def __init__(self, config, db_manager, log_callback=None):
        self.config = config
        self.db = db_manager
        self.log = log_callback or (lambda *_args, **_kwargs: None)

        self._logger = get_component_logger(__name__, "agent_master")
        self._tv_logger = get_component_logger(__name__, "tv_webhook")

        self.mode = self._read_mode()
        self.bus = EventBus()
        self.gov = Governance(config)
        db_dir = getattr(db_manager, "db_dir", None) or os.path.join(os.getcwd(), "db")
        self.metrics = MetricsStore(db_dir)
        self.scheduler = JobScheduler(log_callback=self.log, metrics_store=self.metrics)

        # TradingView alert autovalidation (PAPER-only; gated internally)
        def _tv_autoval_log(msg: str) -> None:
            try:
                self._tv_logger.info(
                    msg,
                    extra={
                        "component": "tv_autovalidation",
                        "incident": "TV_AUTOVAL_MSG",
                    },
                )
            except Exception:
                pass
            try:
                self.log(str(msg))
            except Exception:
                pass

        self._tv_autovalidation = TradingViewAutoValidator(
            self.config,
            self.db,
            self.metrics,
            _tv_autoval_log,
            lambda: self.mode,
        )

        self._tv_server = None

        # TradingView candidate de-dup (per symbol/timeframe/signal)
        self._tv_candidate_last = {}

        self.bus.subscribe("*", self._on_event)
        self._register_jobs()
        self.scheduler.start()

        # Phase 1.5: TradingView webhook ingestion (optional)
        self._start_tradingview_webhook_if_enabled()

    def shutdown(self):
        try:
            if self._tv_server is not None:
                self._tv_server.stop()
        except Exception:
            pass

        try:
            self.scheduler.stop()
        except Exception:
            pass

    def set_mode(self, mode: str):
        candidate = str(mode or "OFF").strip().upper()
        if candidate not in self.MODES:
            return False
        self.mode = candidate
        try:
            self.config["CONFIGURATION"]["agent_mode"] = candidate
        except Exception:
            pass
        self.log(f"🧠 [AGENT] Mode set to {candidate}")
        return True

    def publish(self, event_type: str, payload: dict | None = None):
        self.bus.publish(event_type, payload or {})

    def evaluate_action(self, action: dict):
        action_type = str(action.get("type", "UNKNOWN"))
        scope = "LIVE" if self.mode == "LIVE" else "PAPER"
        action["scope"] = scope

        # OFF/ADVISORY must never block the core engine loop.
        if self.mode == "OFF":
            self.metrics.log_agent_action(self.mode, action_type, True, "Agent mode OFF (bypass)", action)
            return True, "Agent mode OFF"

        if self.mode == "ADVISORY":
            self.metrics.log_agent_action(self.mode, action_type, True, "Advisory-only (bypass)", action)
            return True, "Advisory-only"

        approved, reason = self.gov.approve_action(action)
        self.metrics.log_agent_action(self.mode, action_type, approved, reason, action)
        return approved, reason

    def _register_jobs(self):
        self.scheduler.add_job("agent_health_snapshot", 60, self._health_snapshot)
        if getattr(self, "_tv_autovalidation", None) is not None:
            self.scheduler.add_job("tv_autovalidation_pump", 2, self._tv_autovalidation.pump)

    def _health_snapshot(self):
        score = 1.0
        if self.mode == "OFF":
            score = 0.5
        elif self.mode == "ADVISORY":
            score = 0.7
        elif self.mode == "PAPER":
            score = 0.85
        self.metrics.log_metric("system_health_score", score, {"mode": self.mode})

        # v5.17.0: persist basic symbol/system health snapshot for dashboard widget.
        try:
            stale_threshold = float(self.config.get("CONFIGURATION", "stale_bar_seconds_threshold", fallback="180"))
        except Exception:
            stale_threshold = 180.0

        api_err = 0
        rejects = 0
        total_orders = 0
        try:
            eng = getattr(self.db, "engine", None)
            api_err = int(len(getattr(eng, "_e5_api_error_events", []) or [])) if eng is not None else 0
            rejects = int(len(getattr(eng, "_e5_reject_events", []) or [])) if eng is not None else 0
            total_orders = int(len(getattr(eng, "pending_confirmations", {}) or {})) + rejects
        except Exception:
            api_err, rejects, total_orders = 0, 0, 0

        freshness = None
        try:
            syms = []
            get_syms = getattr(self.db, "get_distinct_symbols", None)
            if callable(get_syms):
                syms = list(get_syms() or [])[:25]
            ages = []
            for sym in syms:
                ts = self.db.get_last_timestamp(sym)
                if ts is None:
                    continue
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                ages.append(float(age))
            if ages:
                freshness = max(ages)
        except Exception:
            freshness = None

        reject_ratio = (float(rejects) / float(total_orders)) if total_orders > 0 else 0.0
        self.metrics.log_symbol_health(
            "SYSTEM",
            freshness_seconds=freshness,
            api_error_streak=api_err,
            reject_ratio=reject_ratio,
            decision_exec_latency_ms=None,
            slippage_bps=None,
            details={"mode": self.mode, "stale_threshold": stale_threshold},
        )

    def _read_mode(self):
        try:
            m = str(self.config.get("CONFIGURATION", "agent_mode", fallback="OFF")).strip().upper()
        except Exception:
            m = "OFF"
        return m if m in self.MODES else "OFF"

    # ------------------------------
    # TradingView webhook ingestion
    # ------------------------------

    def _cfg_bool(self, section: str, key: str, default: bool = False) -> bool:
        try:
            v = str(self.config.get(section, key, fallback=str(default))).strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        except Exception:
            return bool(default)

    def _cfg_int(self, section: str, key: str, default: int) -> int:
        try:
            return int(str(self.config.get(section, key, fallback=str(default))).strip())
        except Exception:
            return int(default)

    def _cfg_str(self, section: str, key: str, default: str = "") -> str:
        try:
            return str(self.config.get(section, key, fallback=default)).strip()
        except Exception:
            return str(default)

    def _start_tradingview_webhook_if_enabled(self) -> None:
        try:
            enabled = self._cfg_bool("TRADINGVIEW", "enabled", False)
            mode = self._cfg_str("TRADINGVIEW", "mode", "ADVISORY").upper()
            if not enabled or mode == "OFF":
                return

            host = self._cfg_str("TRADINGVIEW", "listen_host", "127.0.0.1")
            port = self._cfg_int("TRADINGVIEW", "listen_port", 5001)
            # Secrets live in keys.ini (KEYS.tradingview_secret).
            # Fallback to config.ini [TRADINGVIEW].secret for one-way migration safety.
            secret = self._cfg_str("KEYS", "tradingview_secret", "")
            if not secret:
                secret = self._cfg_str("TRADINGVIEW", "secret", "")
            allow_raw = self._cfg_str("TRADINGVIEW", "allowed_signals", "")
            allowed_signals = [s.strip().upper() for s in allow_raw.split(",") if s.strip()]

            from .integrations.tradingview_webhook import TradingViewWebhookServer

            self._tv_server = TradingViewWebhookServer(
                host,
                port,
                secret=secret,
                allowed_signals=allowed_signals,
                on_alert=self._on_tradingview_alert,
                logger=self._tv_logger,
            )
            self._tv_server.start()
            self.log(f"📡 [TV] Webhook listener ON ({host}:{port})")
        except Exception:
            self._logger.exception("[E_TV_START_FAIL] Failed starting TradingView webhook listener")

    def _on_tradingview_alert(
        self,
        payload: Dict[str, Any],
        raw_json: str,
        idempotency_key: str,
        headers: Dict[str, str],
        client_ip: str,
    ) -> None:
        # Extract common fields (best-effort)
        ts = None
        for k in ("ts", "timestamp", "time", "t"):
            if k in payload:
                ts = payload.get(k)
                break
        try:
            ts_i = int(ts) if ts is not None else int(time.time())
        except Exception:
            ts_i = int(time.time())

        symbol = payload.get("symbol") or payload.get("ticker") or payload.get("sym")
        exchange = payload.get("exchange") or payload.get("ex")
        timeframe = payload.get("timeframe") or payload.get("tf") or payload.get("interval")
        signal = payload.get("signal") or payload.get("action") or payload.get("side")

        price = payload.get("price")
        if price is None:
            price = payload.get("close")
        try:
            price_f: Optional[float] = float(price) if price is not None else None
        except Exception:
            price_f = None

        processed = 1 if bool(payload.get("_ignored")) else 0

        try:
            rid = self.metrics.log_tradingview_alert(
                ts=ts_i,
                symbol=str(symbol).strip() if symbol is not None else None,
                exchange=str(exchange).strip() if exchange is not None else None,
                timeframe=str(timeframe).strip() if timeframe is not None else None,
                signal=str(signal).strip() if signal is not None else None,
                price=price_f,
                raw_json=raw_json,
                idempotency_key=idempotency_key,
                processed=processed,
                extra={"client_ip": client_ip, "headers": headers},
            )
        except Exception:
            self._logger.exception(
                "[E_TV_DB_WRITE_FAIL] Failed persisting TradingView alert",
                extra={
                    "component": "tv_webhook",
                    "symbol": (str(symbol).strip().upper() if symbol else "-"),
                    "mode": self.mode,
                },
            )
            return

        try:
            # Forward-compatible: publish to the bus for downstream candidate pipeline (v5.14.6+)
            self.bus.publish(
                "TRADINGVIEW_ALERT",
                {
                    "metrics_rowid": rid,
                    "ts": ts_i,
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "signal": signal,
                    "price": price_f,
                    "idempotency_key": idempotency_key,
                    "client_ip": client_ip,
                },
            )
        except Exception:
            pass

        try:
            s_sym = str(symbol or "").strip().upper()
            s_sig = str(signal or "").strip().upper()
            self._tv_logger.info(
                f"[TV] Alert received signal={s_sig} symbol={s_sym}",
                extra={"symbol": s_sym or "-", "mode": self.mode, "component": "tv_webhook"},
            )
        except Exception:
            pass

        # Optional UI log line (keep concise)
        try:
            if symbol and signal:
                self.log(f"📡 [TV] {str(signal).strip().upper()} {str(symbol).strip().upper()}")
        except Exception:
            pass

    def _on_tradingview_candidate(self, payload: Dict[str, Any]) -> None:
        """Convert a TradingView alert into a DB candidate (ADVISORY-safe).

        - Stores+scores candidates only (no trading actions).
        - Dedups identical alerts within a cooldown window (symbol/timeframe/signal).
        """
        try:
            if not isinstance(payload, dict):
                return

            if payload.get("_ignored"):
                return

            mode = (self._cfg_str("TRADINGVIEW", "mode", "ADVISORY") or "ADVISORY").upper()
            if mode == "OFF":
                return

            symbol = str(payload.get("symbol") or "").strip().upper()
            if not symbol:
                return

            timeframe = str(payload.get("timeframe") or "").strip()
            signal = str(payload.get("signal") or "").strip().upper()
            exchange = str(payload.get("exchange") or "").strip().upper()
            price = payload.get("price")
            idem = str(payload.get("idempotency_key") or "").strip()

            cooldown_min = max(0, int(self._cfg_int("TRADINGVIEW", "candidate_cooldown_minutes", 5)))
            now_ts = int(payload.get("ts") or time.time())
            key = (symbol, timeframe, signal)
            last = self._tv_candidate_last.get(key)
            if cooldown_min > 0 and last and (now_ts - int(last)) < (cooldown_min * 60):
                self._tv_logger.info(
                    "TV candidate dedup hit",
                    extra={
                        "component": "tv_candidate",
                        "incident": "TV_CAND_DEDUP",
                        "symbol": symbol,
                        "mode": mode,
                    },
                )
                return
            self._tv_candidate_last[key] = now_ts

            from .research.candidate_scanner import CandidateScanner

            scanner = CandidateScanner(self.db, self.config, log=self._logger)
            extra_details = {
                "source": "TRADINGVIEW",
                "tv_timeframe": timeframe,
                "tv_signal": signal,
                "tv_exchange": exchange,
                "tv_price": price,
                "tv_idempotency_key": idem,
                "tv_alert_ts": now_ts,
            }

            row = scanner.score_single_symbol(
                symbol,
                universe="TRADINGVIEW",
                extra_details=extra_details,
                force_accept=True,
            )
            if not row:
                return

            scan_date = row.get("scan_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            sid = f"TV_{scan_date}"
            if timeframe:
                sid += f"_{timeframe}"
            if signal:
                sid += f"_{signal}"
            sid = re.sub(r"[^A-Za-z0-9_\-]", "_", sid)

            self.db.save_candidates(sid, [row])

            # PAPER-only: enqueue autovalidation (backfill + quick sanity backtest) if enabled
            try:
                if getattr(self, "_tv_autovalidation", None) is not None:
                    self._tv_autovalidation.maybe_enqueue(payload)
            except Exception:
                self._tv_logger.exception(
                    "TV autovalidation enqueue failed",
                    extra={"component": "tv_autovalidation", "incident": "TV_AUTOVAL_ENQ_FAIL"},
                )

            self._tv_logger.info(
                "TV candidate saved",
                extra={
                    "component": "tv_candidate",
                    "incident": "TV_CAND_SAVED",
                    "symbol": symbol,
                    "mode": mode,
                },
            )
        except Exception:
            self._tv_logger.exception(
                "TV candidate conversion failed",
                extra={
                    "component": "tv_candidate",
                    "incident": "TV_CAND_FAIL",
                },
            )

    # ------------------------------
    # Event bus
    # ------------------------------

    def _on_event(self, event: dict):
        ev_type = str(event.get("type", ""))
        payload = event.get("payload", {}) or {}

        if ev_type in {"ORDER_REJECTED", "RISK_BREACH", "DATA_GAP"}:
            self.metrics.log_metric("anomaly_count", 1.0, {"event_type": ev_type, **payload})
            self.metrics.log_anomaly(ev_type, severity="WARN", source="event_bus", details=payload)
            if self.mode in {"PAPER", "LIVE"}:
                self.log(f"🧠 [AGENT] Detected {ev_type}; tightening guardrails.")

        if ev_type == "TRADINGVIEW_ALERT":
            self._on_tradingview_candidate(payload)
            return

        if ev_type == "ACTION_REQUEST":
            approved, reason = self.evaluate_action(payload)
            self.log(
                f"🧠 [AGENT] Action {payload.get('type', 'UNKNOWN')}: {'APPROVED' if approved else 'DENIED'} ({reason})"
            )