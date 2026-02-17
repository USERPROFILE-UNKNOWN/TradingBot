"""AI Master Agent orchestration with guarded paper/live controls."""

from __future__ import annotations

import os
import re
import json
import sqlite3
import hashlib
import time
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .event_bus import EventBus
from .scheduler import JobScheduler
from .governance import Governance
from .metrics import MetricsStore
from .experiments import ExperimentsStore
from .logging_utils import get_component_logger
from .research.tv_autovalidation import TradingViewAutoValidator
from .watchlist_api import add_watchlist_symbol, get_watchlist_symbols, remove_watchlist_symbol


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
        self.experiments = ExperimentsStore(db_dir)
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
        self._last_backfill_started_at = 0.0
        self._last_daily_report_date = ""
        self._last_research_sweep_date = ""
        self._live_change_day = ""
        self._live_config_tunes_today = 0
        self._live_promotions_today = 0
        self._hard_halt_active = False

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

        # v5.20.0: hard-halt supremacy and bounded autonomy counters
        if self._cfg_bool("CONFIGURATION", "agent_hard_halt_supreme", True) and self._hard_halt_active:
            if action_type not in {"CLEAR_HARD_HALT", "HALT_ACK"}:
                self.metrics.log_agent_action(self.mode, action_type, False, "Hard halt active", action)
                return False, "Hard halt active"

        approved, reason = self.gov.approve_action(action)
        if approved and self.mode == "LIVE":
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if day != self._live_change_day:
                self._live_change_day = day
                self._live_config_tunes_today = 0
                self._live_promotions_today = 0

            if action_type == "CONFIG_CHANGE":
                max_tunes = max(0, self._cfg_int("CONFIGURATION", "agent_max_config_tunes_per_day", 2))
                if self._live_config_tunes_today >= max_tunes:
                    approved, reason = False, "Config tuning daily limit reached"
                else:
                    self._live_config_tunes_today += 1
            elif action_type == "DEPLOY_STRATEGY":
                max_promos = max(0, self._cfg_int("CONFIGURATION", "agent_max_promotions_per_day", 1))
                if self._live_promotions_today >= max_promos:
                    approved, reason = False, "Promotion daily limit reached"
                else:
                    self._live_promotions_today += 1

        self.metrics.log_agent_action(self.mode, action_type, approved, reason, action)
        return approved, reason

    def _register_jobs(self):
        self.scheduler.add_job("agent_health_snapshot", 60, self._health_snapshot)
        if self._cfg_bool("CONFIGURATION", "agent_db_integrity_check_enabled", True):
            self.scheduler.add_job("agent_db_integrity_check", 900, self._run_db_integrity_check)
        if self._cfg_bool("CONFIGURATION", "agent_stale_quarantine_enabled", True):
            self.scheduler.add_job("agent_stale_quarantine", 600, self._run_stale_symbol_quarantine)
        if self._cfg_bool("CONFIGURATION", "agent_auto_backfill_enabled", True):
            self.scheduler.add_job("agent_auto_backfill", 300, self._run_auto_backfill)
        if self._cfg_bool("CONFIGURATION", "agent_daily_report_enabled", True):
            self.scheduler.add_job("agent_daily_report", 600, self._run_daily_report)
        if self._cfg_bool("CONFIGURATION", "agent_research_automation_enabled", True):
            self.scheduler.add_job("agent_research_sweep", 900, self._run_research_sweep)
        if self._cfg_bool("CONFIGURATION", "agent_canary_enabled", True):
            self.scheduler.add_job("agent_live_canary_guardrails", 60, self._run_live_canary_guardrails)
        if getattr(self, "_tv_autovalidation", None) is not None:
            self.scheduler.add_job("tv_autovalidation_pump", 2, self._tv_autovalidation.pump)

    def _run_db_integrity_check(self) -> None:
        db_paths = []
        try:
            if getattr(self.db, "db_paths", None):
                db_paths.extend(str(v) for v in self.db.db_paths.values() if v)
        except Exception:
            db_paths = []

        if not db_paths:
            try:
                db_dir = getattr(self.db, "db_dir", "")
                if db_dir and os.path.isdir(db_dir):
                    db_paths = [os.path.join(db_dir, n) for n in os.listdir(db_dir) if n.lower().endswith(".db")]
            except Exception:
                db_paths = []

        if not db_paths:
            return

        checked = 0
        issues = 0
        for p in db_paths:
            if not p or not os.path.exists(p):
                continue
            checked += 1
            try:
                with sqlite3.connect(p, timeout=5) as conn:
                    row = conn.execute("PRAGMA quick_check").fetchone()
                    conn.execute("PRAGMA optimize")
                ok = str((row or [""])[0]).strip().lower() == "ok"
                if not ok:
                    issues += 1
                    self.metrics.log_anomaly(
                        "DB_INTEGRITY_FAIL",
                        severity="ERROR",
                        source="agent_maintenance",
                        details={"db_path": p, "quick_check": (row[0] if row else "")},
                    )
            except Exception as e:
                issues += 1
                self.metrics.log_anomaly(
                    "DB_INTEGRITY_ERROR",
                    severity="ERROR",
                    source="agent_maintenance",
                    details={"db_path": p, "error": str(e)},
                )

        self.metrics.log_metric(
            "agent_db_integrity_issues",
            float(issues),
            {"checked": checked},
        )

    def _run_stale_symbol_quarantine(self) -> None:
        threshold = float(self._cfg_int("CONFIGURATION", "agent_stale_quarantine_threshold_seconds", 21600))
        if threshold <= 0:
            return

        active = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL") or [])
        if not active:
            return

        stale_syms = []
        now = datetime.now(timezone.utc)
        for sym in active:
            try:
                ts = self.db.get_last_timestamp(sym)
            except Exception:
                ts = None
            if ts is None:
                stale_syms.append(sym)
                continue
            try:
                age = (now - ts).total_seconds()
            except Exception:
                age = threshold + 1
            if age >= threshold:
                stale_syms.append(sym)

        moved = []
        for sym in stale_syms:
            removed = remove_watchlist_symbol(self.config, sym, group="ACTIVE")
            if removed and add_watchlist_symbol(self.config, sym, group="ARCHIVE"):
                moved.append(sym)

        if moved:
            self._persist_split_config()
            self.metrics.log_anomaly(
                "STALE_SYMBOL_QUARANTINE",
                severity="WARN",
                source="agent_maintenance",
                details={"count": len(moved), "symbols": moved[:25]},
            )
            self.log(f"🧠 [AGENT] Quarantined {len(moved)} stale symbols to ARCHIVE watchlist.")

    def _run_auto_backfill(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return
        min_interval_min = max(1, self._cfg_int("CONFIGURATION", "agent_backfill_min_interval_minutes", 240))
        now = time.time()
        if (now - float(self._last_backfill_started_at)) < (min_interval_min * 60):
            return

        key = self._cfg_str("KEYS", "alpaca_key", "")
        secret = self._cfg_str("KEYS", "alpaca_secret", "")
        if not key or not secret:
            return

        self._last_backfill_started_at = now
        try:
            from .backfill import BackfillEngine

            be = BackfillEngine(self.config, self.db, self.log)
            be.run()
        except Exception as e:
            self.metrics.log_anomaly(
                "AUTO_BACKFILL_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

    def _run_daily_report(self) -> None:
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day == self._last_daily_report_date:
            return

        if self._cfg_int("CONFIGURATION", "agent_daily_report_hour_utc", 0) != now.hour:
            return

        lookback_sec = 24 * 3600
        snap = self.metrics.get_health_widget_snapshot(lookback_sec=lookback_sec)
        actions = self.metrics.latest_actions(limit=20)
        payload = {
            "date": day,
            "mode": self.mode,
            "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "health": snap,
            "recent_actions": actions,
        }

        logs_dir = self._paths_get("logs", os.path.join(os.getcwd(), "logs"))
        out_dir = os.path.join(logs_dir, "summaries")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"agent_daily_{day}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self._last_daily_report_date = day
        self.metrics.log_metric("agent_daily_report_written", 1.0, {"path": out_path})

    def _classify_regime(self, symbol: str) -> str:
        try:
            df = self.db.get_history(symbol, limit=240)
            if df is None or len(df) < 50:
                return "UNKNOWN"
            close = df["close"].astype(float)
            atr = df["atr"].astype(float) if "atr" in df.columns else None
            ema = df["ema_200"].astype(float) if "ema_200" in df.columns else None
            c = float(close.iloc[-1])
            if c <= 0:
                return "UNKNOWN"

            if atr is not None and len(atr) > 0:
                atr_pct = float(atr.iloc[-1]) / c if float(atr.iloc[-1]) > 0 else 0.0
                highvol_min = float(self._cfg_str("CONFIGURATION", "regime_highvol_atr_pct_min", "0.03") or "0.03")
                if atr_pct >= highvol_min:
                    return "HIGH_VOL"

            if ema is not None and len(ema) > 0:
                e = float(ema.iloc[-1])
                if c > e:
                    return "BULL"
                if c < e:
                    return "BEAR"
            return "CHOP"
        except Exception:
            return "UNKNOWN"

    def _run_research_sweep(self) -> None:
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day == self._last_research_sweep_date:
            return

        if self._cfg_int("CONFIGURATION", "agent_research_sweep_hour_utc", 1) != now.hour:
            return

        if self.mode == "OFF":
            return

        symbols = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL") or [])
        max_symbols = max(1, self._cfg_int("CONFIGURATION", "agent_research_max_symbols", 10))
        symbols = symbols[:max_symbols]
        if not symbols:
            return

        try:
            from .research.quick_backtest import run_quick_backtest
        except Exception as e:
            self.metrics.log_anomaly("RESEARCH_SWEEP_IMPORT_FAIL", source="agent_research", details={"error": str(e)})
            return

        bt_days = max(1, self._cfg_int("CONFIGURATION", "agent_research_backtest_days", 14))
        max_strats = max(1, self._cfg_int("CONFIGURATION", "agent_research_max_strategies", 6))
        min_trades = max(0, self._cfg_int("CONFIGURATION", "agent_research_min_trades", 1))

        cfg_hash = self._config_snapshot_hash()
        cfg_json = self._config_snapshot_json()
        try:
            if cfg_hash:
                self.experiments.upsert_config_snapshot(
                    snapshot_hash=cfg_hash,
                    config_json=cfg_json,
                    source="agent_research_sweep",
                    note="Daily research automation sweep",
                )
        except Exception:
            self._logger.exception("[E_EXPERIMENT_CFG_SNAPSHOT] Failed to write config snapshot")

        out = []
        ok_count = 0
        for sym in symbols:
            regime = self._classify_regime(sym)
            res = run_quick_backtest(
                self.config,
                self.db,
                sym,
                days=bt_days,
                max_strategies=max_strats,
                min_trades=min_trades,
            )
            row = {"symbol": sym, "regime": regime, "result": res}
            out.append(row)

            try:
                best = (res or {}).get("best") or {}
                strat_name = str(best.get("strategy") or "")
                total_pl = float(best.get("total_pl") or 0.0) if best else 0.0

                self.experiments.log_experiment(
                    symbol=sym,
                    regime=regime,
                    hypothesis="daily_quick_backtest_sweep",
                    strategy_name=strat_name,
                    score=total_pl,
                    status=("paper_candidate" if bool((res or {}).get("ok")) else "failed"),
                    details={"result": res},
                    config_hash=cfg_hash,
                )

                if strat_name:
                    self.experiments.upsert_strategy(
                        strategy_name=strat_name,
                        strategy_version="v5.19.2",
                        status=("paper_candidate" if bool((res or {}).get("ok")) else "failed"),
                        params={},
                        metadata={"symbol": sym, "regime": regime, "source": "agent_research_sweep"},
                    )
                    self.experiments.log_deployment(
                        strategy_name=strat_name,
                        strategy_version="v5.19.2",
                        stage="PAPER",
                        status=("candidate" if bool((res or {}).get("ok")) else "failed"),
                        reason="daily_research_sweep",
                        metrics={"total_pl": total_pl, "symbol": sym, "regime": regime},
                    )
            except Exception:
                self._logger.exception("[E_EXPERIMENT_LOG] Failed persisting experiment rows")

            if bool(res.get("ok")):
                ok_count += 1

        logs_dir = self._paths_get("logs", os.path.join(os.getcwd(), "logs"))
        out_dir = os.path.join(logs_dir, "research")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"agent_research_sweep_{day}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": day,
                    "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mode": self.mode,
                    "symbols": len(symbols),
                    "backtest_days": bt_days,
                    "max_strategies": max_strats,
                    "min_trades": min_trades,
                    "ok_results": ok_count,
                    "results": out,
                },
                f,
                indent=2,
            )

        self.metrics.log_metric("agent_research_sweep_symbols", float(len(symbols)), {"ok": ok_count})
        self.metrics.log_metric("agent_research_sweep_written", 1.0, {"path": out_path})
        self._last_research_sweep_date = day

    def _run_live_canary_guardrails(self) -> None:
        if self.mode != "LIVE":
            return

        if not self._cfg_bool("CONFIGURATION", "agent_canary_enabled", True):
            return

        max_reject_pct = float(self._cfg_str("CONFIGURATION", "agent_canary_reject_rate_pct_max", "10.0") or "10.0")
        max_slip_bps = float(self._cfg_str("CONFIGURATION", "agent_canary_slippage_bps_max", "25.0") or "25.0")
        max_drawdown_pct = float(self._cfg_str("CONFIGURATION", "agent_canary_drawdown_pct_max", "4.0") or "4.0")

        snap = self.metrics.get_health_widget_snapshot(lookback_sec=3600)
        reject_pct = float(snap.get("order_reject_ratio") or 0.0) * 100.0
        slippage_bps = float(snap.get("slippage_bps") or 0.0)

        drawdown_pct = 0.0
        try:
            stats = self.db.get_portfolio_stats()
            total_pl = float((stats or {}).get("total_pl") or 0.0)
            base = max(1.0, float(self._cfg_str("CONFIGURATION", "daily_budget", "1000") or "1000"))
            drawdown_pct = max(0.0, abs(min(total_pl, 0.0)) / base * 100.0)
        except Exception:
            drawdown_pct = 0.0

        breaches = []
        if reject_pct > max_reject_pct:
            breaches.append(f"reject_rate={reject_pct:.2f}%>{max_reject_pct:.2f}%")
        if slippage_bps > max_slip_bps:
            breaches.append(f"slippage={slippage_bps:.2f}bps>{max_slip_bps:.2f}bps")
        if drawdown_pct > max_drawdown_pct:
            breaches.append(f"drawdown={drawdown_pct:.2f}%>{max_drawdown_pct:.2f}%")

        if not breaches:
            return

        rollback_mode = self._cfg_str("CONFIGURATION", "agent_canary_rollback_mode", "PAPER").upper()
        if rollback_mode not in self.MODES:
            rollback_mode = "PAPER"

        self.metrics.log_anomaly(
            "CANARY_ROLLBACK_TRIGGERED",
            severity="ERROR",
            source="agent_canary",
            details={
                "breaches": breaches,
                "reject_rate_pct": reject_pct,
                "slippage_bps": slippage_bps,
                "drawdown_pct": drawdown_pct,
                "rollback_mode": rollback_mode,
            },
        )
        self.set_mode(rollback_mode)
        self.log(f"🛑 [AGENT] Canary rollback triggered -> mode={rollback_mode} ({'; '.join(breaches)})")

    def _config_snapshot_hash(self) -> str:
        try:
            sec = {}
            if hasattr(self.config, "has_section") and self.config.has_section("CONFIGURATION"):
                sec = {str(k): str(v) for k, v in self.config.items("CONFIGURATION")}
            payload = json.dumps(sec, sort_keys=True)
            return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""

    def _config_snapshot_json(self) -> dict:
        try:
            if hasattr(self.config, "has_section") and self.config.has_section("CONFIGURATION"):
                return {str(k): str(v) for k, v in self.config.items("CONFIGURATION")}
        except Exception:
            pass
        return {}

    def _paths_get(self, key: str, default: str = "") -> str:
        try:
            p = getattr(self.db, "paths", None)
            if isinstance(p, dict):
                return str(p.get(key, default) or default)
        except Exception:
            pass
        return default

    def _persist_split_config(self) -> None:
        try:
            from .config_io import write_split_config

            p = getattr(self.db, "paths", None)
            if isinstance(p, dict):
                write_split_config(self.config, p)
        except Exception:
            self._logger.exception("[E_AGENT_CFG_WRITE] Failed persisting split config")

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
            if ev_type == "RISK_BREACH" and self._cfg_bool("CONFIGURATION", "agent_hard_halt_supreme", True):
                self._hard_halt_active = True
            if self.mode in {"PAPER", "LIVE"}:
                self.log(f"🧠 [AGENT] Detected {ev_type}; tightening guardrails.")

        if ev_type == "TRADINGVIEW_ALERT":
            self._on_tradingview_candidate(payload)
            return

        if ev_type == "ACTION_REQUEST":
            at = str((payload or {}).get("type", "")).upper()
            if at == "CLEAR_HARD_HALT":
                self._hard_halt_active = False
                self.metrics.log_anomaly("HARD_HALT_CLEARED", severity="INFO", source="event_bus", details=payload)
                self.log("🧠 [AGENT] Hard halt cleared by action request.")
                return
            approved, reason = self.evaluate_action(payload)
            self.log(
                f"🧠 [AGENT] Action {payload.get('type', 'UNKNOWN')}: {'APPROVED' if approved else 'DENIED'} ({reason})"
            )
