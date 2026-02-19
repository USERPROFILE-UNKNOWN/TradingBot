"""AI Master Agent orchestration with guarded paper/live controls."""

from __future__ import annotations

import os
import re
import json
import sqlite3
import hashlib
import time
import threading
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .event_bus import EventBus
from .scheduler import JobScheduler
from .governance import Governance
from .metrics import MetricsStore
from .experiments import ExperimentsStore
from .logging_utils import get_component_logger
from .research.tv_autovalidation import TradingViewAutoValidator
from .research.watchlist_policy import apply_watchlist_policy
from .research.full_backtest_service import run_full_backtest_service
from .watchlist_api import get_watchlist_symbols
from .agent_shadow import AgentShadow


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
        self._shadow_agent = None

        # TradingView candidate de-dup (per symbol/timeframe/signal)
        self._tv_candidate_last = {}
        self._agent_started_at = time.time()
        self._last_backfill_started_at = 0.0
        self._last_daily_report_date = ""
        self._last_research_sweep_date = ""
        self._last_full_backtest_date = ""
        self._live_change_day = ""
        self._live_config_tunes_today = 0
        self._live_promotions_today = 0
        self._hard_halt_active = False
        self._autopilot_seq = 0

        self.bus.subscribe("*", self._on_event)
        self._started = False
        self._state_write_lock = threading.RLock()
        self._engine_running_provider = None


    def start(self) -> bool:
        """Start optional automation services (explicit opt-in)."""
        if bool(getattr(self, "_started", False)):
            return False
        self._register_jobs()
        self.scheduler.start()

        # Phase 1.5: TradingView webhook ingestion (optional)
        self._start_tradingview_webhook_if_enabled()

        # v6.20.0: AgentShadow proposal loop (PAPER-first, no direct execution).
        self._start_agent_shadow_if_enabled()
        self._started = True
        return True

    def attach_state_controls(self, *, state_write_lock=None, engine_running_provider=None) -> None:
        """Attach shared mutation lock and optional engine-running provider."""
        if state_write_lock is not None:
            self._state_write_lock = state_write_lock
        self._engine_running_provider = engine_running_provider

    def _engine_running(self) -> bool:
        fn = getattr(self, "_engine_running_provider", None)
        if fn is None:
            return True
        try:
            return bool(fn())
        except Exception:
            return False

    def shutdown(self):
        try:
            if self._tv_server is not None:
                self._tv_server.stop()
        except Exception:
            pass

        try:
            if bool(getattr(self, "_started", False)):
                self.scheduler.stop()
        except Exception:
            pass

        try:
            if self._shadow_agent is not None:
                self._shadow_agent.stop()
        except Exception:
            pass
        self._started = False

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

    def _resolve_runtime_paths(self) -> Dict[str, str]:
        paths = {}
        try:
            raw = getattr(self.db, "paths", None)
            if isinstance(raw, dict):
                paths.update({str(k): str(v) for k, v in raw.items() if v})
        except Exception:
            pass

        if not paths.get("logs"):
            paths["logs"] = os.path.join(os.getcwd(), "logs")
        return paths

    def _start_agent_shadow_if_enabled(self) -> None:
        try:
            enabled = self._cfg_bool("CONFIGURATION", "agent_shadow_enabled", True)
        except Exception:
            enabled = True

        if not enabled:
            return

        try:
            paths = self._resolve_runtime_paths()
            self._shadow_agent = AgentShadow(self.config, self.db, paths, log_fn=self.log)
            self._shadow_agent.start()
            self.log("🧠 [AGENT] AgentShadow started.")
        except Exception as e:
            self.log(f"[AGENT] Shadow startup failed: {e}")

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

        if self.mode == "LIVE" and action_type == "DEPLOY_STRATEGY":
            gates_ok, gate_reason = self._check_live_promotion_gates(action)
            if not gates_ok:
                self.metrics.log_agent_action(self.mode, action_type, False, gate_reason, action)
                return False, gate_reason

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

    def _check_live_promotion_gates(self, action: dict) -> tuple[bool, str]:
        """Hard-stop promotion gates for LIVE deploy actions."""
        if not self._cfg_bool("CONFIGURATION", "promotion_enabled", True):
            return False, "Promotion disabled"

        stats = {}
        try:
            stats = dict(action.get("promotion_stats") or {})
        except Exception:
            stats = {}

        def _num(key: str, default: float = 0.0) -> float:
            try:
                if key in action and action.get(key) is not None:
                    return float(action.get(key))
                return float(stats.get(key, default))
            except Exception:
                return float(default)

        sessions = int(_num("paper_sessions", 0))
        trades = int(_num("paper_trades_total", 0))
        drawdown_pct = _num("drawdown_pct", 100.0)
        cancel_rate_pct = _num("cancel_rate_pct", 100.0)
        reject_rate_pct = _num("reject_rate_pct", 100.0)
        watchdog_halts = int(_num("watchdog_halts", 0))
        stale_symbols = int(_num("stale_symbols", 0))

        min_sessions = self._cfg_int("CONFIGURATION", "promotion_min_sessions", 5)
        min_trades = self._cfg_int("CONFIGURATION", "promotion_min_trades_total", 30)
        max_drawdown = self._cfg_float("CONFIGURATION", "promotion_max_drawdown_pct", 4.0)
        max_cancel = self._cfg_float("CONFIGURATION", "promotion_max_cancel_rate_pct", 35.0)
        max_reject = self._cfg_float("CONFIGURATION", "promotion_max_reject_rate_pct", 10.0)

        if sessions < min_sessions:
            return False, f"Promotion gate failed: sessions {sessions} < {min_sessions}"
        if trades < min_trades:
            return False, f"Promotion gate failed: trades {trades} < {min_trades}"
        if drawdown_pct > max_drawdown:
            return False, f"Promotion gate failed: drawdown {drawdown_pct:.2f}% > {max_drawdown:.2f}%"
        if cancel_rate_pct > max_cancel:
            return False, f"Promotion gate failed: cancel rate {cancel_rate_pct:.2f}% > {max_cancel:.2f}%"
        if reject_rate_pct > max_reject:
            return False, f"Promotion gate failed: reject rate {reject_rate_pct:.2f}% > {max_reject:.2f}%"

        if self._cfg_bool("CONFIGURATION", "promotion_require_no_watchdog_halts", True) and watchdog_halts > 0:
            return False, "Promotion gate failed: watchdog halts present"
        if self._cfg_bool("CONFIGURATION", "promotion_require_no_stale_symbols", True) and stale_symbols > 0:
            return False, "Promotion gate failed: stale symbols present"
        if self._cfg_bool("CONFIGURATION", "agent_hard_halt_supreme", True) and bool(getattr(self, "_hard_halt_active", False)):
            return False, "Promotion gate failed: hard halt active"

        return True, "Promotion gates passed"

    def _register_jobs(self):
        self.scheduler.add_job("agent_health_snapshot", 60, self._health_snapshot)
        if self._cfg_bool("CONFIGURATION", "agent_db_integrity_check_enabled", True):
            self.scheduler.add_job("agent_db_integrity_check", 900, self._run_db_integrity_check)
        if self._cfg_bool("CONFIGURATION", "agent_stale_quarantine_enabled", True):
            self.scheduler.add_job("agent_stale_quarantine", 600, self._run_stale_symbol_quarantine)
        if self._cfg_bool("CONFIGURATION", "agent_auto_backfill_enabled", True):
            self.scheduler.add_job("agent_auto_backfill", 300, self._run_auto_backfill)
        if self._cfg_bool("CONFIGURATION", "agent_candidate_scan_enabled", True):
            scan_min = max(5, self._cfg_int("CONFIGURATION", "agent_candidate_scan_interval_minutes", 60))
            self.scheduler.add_job("agent_candidate_scan", scan_min * 60, self._run_candidate_scan)
        if self._cfg_bool("CONFIGURATION", "agent_candidate_simulation_enabled", True):
            sim_min = max(5, self._cfg_int("CONFIGURATION", "agent_candidate_simulation_interval_minutes", 60))
            self.scheduler.add_job("agent_candidate_simulation", sim_min * 60, self._run_candidate_simulation)
        if self._cfg_bool("CONFIGURATION", "agent_watchlist_policy_enabled", True):
            wl_min = max(5, self._cfg_int("CONFIGURATION", "agent_watchlist_policy_interval_minutes", 60))
            self.scheduler.add_job("agent_watchlist_policy_update", wl_min * 60, self._run_watchlist_policy_update)
        if self._cfg_bool("CONFIGURATION", "agent_quick_backtest_enabled", True):
            qb_min = max(5, self._cfg_int("CONFIGURATION", "agent_quick_backtest_interval_minutes", 1440))
            self.scheduler.add_job("agent_quick_backtests", qb_min * 60, self._run_quick_backtests)
        if self._cfg_bool("CONFIGURATION", "agent_architect_optimize_enabled", True):
            ao_min = max(15, self._cfg_int("CONFIGURATION", "agent_architect_optimize_interval_minutes", 10080))
            self.scheduler.add_job("agent_architect_optimize", ao_min * 60, self._run_architect_optimize)
        if self._cfg_bool("CONFIGURATION", "agent_architect_orchestrator_enabled", True):
            orch_min = max(15, self._cfg_int("CONFIGURATION", "agent_architect_orchestrator_interval_minutes", 10080))
            self.scheduler.add_job("agent_architect_orchestrator", orch_min * 60, self._run_architect_orchestrator)
        if self._cfg_bool("CONFIGURATION", "agent_full_backtest_enabled", False):
            fb_min = max(15, self._cfg_int("CONFIGURATION", "agent_full_backtest_interval_minutes", 1440))
            self.scheduler.add_job("agent_full_backtests", fb_min * 60, self._run_full_backtests)
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

    def _is_equity_market_hours_utc(self, now_utc: datetime) -> bool:
        """Best-effort weekday market-hours gate in UTC."""
        try:
            if int(now_utc.weekday()) >= 5:
                return False
        except Exception:
            return False

        start_h = max(0, min(23, self._cfg_int("CONFIGURATION", "agent_stale_quarantine_equity_market_open_hour_utc", 14)))
        end_h = max(0, min(23, self._cfg_int("CONFIGURATION", "agent_stale_quarantine_equity_market_close_hour_utc", 21)))
        h = int(getattr(now_utc, "hour", 0) or 0)

        if start_h <= end_h:
            return start_h <= h < end_h
        return h >= start_h or h < end_h

    def _stale_quarantine_daily_budget_remaining(self, now_utc: datetime) -> int:
        budget = max(0, self._cfg_int("CONFIGURATION", "agent_stale_quarantine_max_per_day", 12))
        day_key = now_utc.strftime("%Y-%m-%d")
        if getattr(self, "_stale_quarantine_day", None) != day_key:
            self._stale_quarantine_day = day_key
            self._stale_quarantine_count_today = 0
        used = int(getattr(self, "_stale_quarantine_count_today", 0) or 0)
        return max(0, int(budget) - used)

    def _run_stale_symbol_quarantine(self) -> None:
        eq_threshold = float(self._cfg_int("CONFIGURATION", "agent_stale_quarantine_threshold_seconds", 21600))
        eq_after_hours_threshold = float(self._cfg_int("CONFIGURATION", "agent_stale_quarantine_equity_after_hours_threshold_seconds", 86400))
        crypto_threshold = float(self._cfg_int("CONFIGURATION", "agent_stale_quarantine_crypto_threshold_seconds", int(eq_threshold)))
        if eq_threshold <= 0 and crypto_threshold <= 0:
            return

        warmup_minutes = max(0, self._cfg_int("CONFIGURATION", "agent_stale_quarantine_warmup_minutes", 45))
        now_epoch = time.time()
        started = float(getattr(self, "_agent_started_at", now_epoch) or now_epoch)
        warmup_remaining = (warmup_minutes * 60.0) - max(0.0, now_epoch - started)
        if warmup_remaining > 0:
            self.metrics.log_metric(
                "agent_stale_quarantine_warmup_remaining_seconds",
                float(warmup_remaining),
                {"warmup_minutes": int(warmup_minutes)},
            )
            return

        cooldown_minutes = max(0, self._cfg_int("CONFIGURATION", "agent_stale_quarantine_cooldown_minutes", 60))
        cooldown_until = float(getattr(self, "_stale_quarantine_cooldown_until", 0.0) or 0.0)
        if cooldown_until > now_epoch:
            self.metrics.log_metric(
                "agent_stale_quarantine_cooldown_remaining_seconds",
                float(cooldown_until - now_epoch),
                {"cooldown_minutes": int(cooldown_minutes)},
            )
            return

        active = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL") or [])
        if not active:
            return

        now = datetime.now(timezone.utc)
        equities_market_hours_only = self._cfg_bool("CONFIGURATION", "agent_stale_quarantine_equity_market_hours_only", True)
        is_equity_hours = AgentMaster._is_equity_market_hours_utc(self, now)

        stale_syms = []
        uninitialized_syms = []

        for sym in active:
            try:
                ts = self.db.get_last_timestamp(sym)
            except Exception:
                ts = None

            if ts is None:
                uninitialized_syms.append(sym)
                continue

            try:
                age = (now - ts).total_seconds()
            except Exception:
                age = max(eq_threshold, crypto_threshold, 1.0) + 1.0

            if "/" in str(sym):
                threshold = crypto_threshold
            else:
                if equities_market_hours_only and (not is_equity_hours):
                    threshold = eq_after_hours_threshold
                else:
                    threshold = eq_threshold

            if threshold > 0 and age >= threshold:
                stale_syms.append(sym)

        if uninitialized_syms and self._cfg_bool("CONFIGURATION", "agent_auto_backfill_enabled", True):
            self.log(
                f"🧠 [AGENT] Detected {len(uninitialized_syms)} uninitialized symbol(s); skipping quarantine and triggering backfill check."
            )
            try:
                self._run_auto_backfill()
            except Exception:
                pass

        budget_left = AgentMaster._stale_quarantine_daily_budget_remaining(self, now)
        if budget_left <= 0:
            self.metrics.log_metric("agent_stale_quarantine_budget_remaining", 0.0, {"budget_exhausted": 1})
            return

        if len(stale_syms) > budget_left:
            stale_syms = stale_syms[:budget_left]

        reported = list(stale_syms)
        if not reported:
            return

        self._stale_quarantine_count_today = int(getattr(self, "_stale_quarantine_count_today", 0) or 0) + len(reported)
        if cooldown_minutes > 0:
            self._stale_quarantine_cooldown_until = now_epoch + (cooldown_minutes * 60.0)
        self.metrics.log_anomaly(
            "STALE_SYMBOL_REPORT",
            severity="WARN",
            source="agent_maintenance",
            details={"count": len(reported), "symbols": reported[:25], "mode": "READ_ONLY"},
        )
        self.log(f"🧠 [AGENT] Stale symbol report generated ({len(reported)} symbol(s)); watchlists unchanged.")

    def _run_candidate_scan(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-scan-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(
                run_id,
                mode=str(self.mode),
                phase="SCAN",
                status="OK",
                summary={"job": "agent_candidate_scan"},
            )
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_candidate_scan", "rows": 0, "scan_id": ""}
        try:
            from .research.candidate_scanner import CandidateScanner

            scanner = CandidateScanner(self.db, self.config, log=self._logger)
            scan_id, rows = scanner.scan_today()
            count = int(len(rows or []))
            top_symbols = [str((r or {}).get("symbol", "")).upper() for r in (rows or [])[:10] if (r or {}).get("symbol")]
            summary.update({"rows": count, "scan_id": str(scan_id), "top_symbols": top_symbols})

            self.metrics.log_metric("agent_candidate_scan_rows", float(count), {"scan_id": str(scan_id), "mode": self.mode})
            self.publish("candidate_scan_completed", {"scan_id": str(scan_id), "count": count, "symbols": top_symbols})

            if count > 0:
                self.log(f"🧠 [AGENT] Candidate scan complete: {count} row(s) (scan_id={scan_id}).")
            else:
                status = "SKIP"
                self.log("🧠 [AGENT] Candidate scan produced no rows.")
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "CANDIDATE_SCAN_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_candidate_simulation(self, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-sim-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(
                run_id,
                mode=str(self.mode),
                phase="SIM",
                status="OK",
                summary={"job": "agent_candidate_simulation"},
            )
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_candidate_simulation", "input": 0, "scored": 0, "scan_id": ""}
        try:
            max_symbols = max(1, self._cfg_int("CONFIGURATION", "agent_candidate_simulation_max_symbols", 10))
            latest = self.db.get_latest_candidates(limit=max_symbols) if hasattr(self.db, "get_latest_candidates") else []
            rows = []
            try:
                if hasattr(latest, "to_dict"):
                    rows = list(latest.to_dict("records"))
                elif isinstance(latest, list):
                    rows = list(latest)
            except Exception:
                rows = []

            symbols = []
            seen = set()
            for r in rows:
                sym = str((r or {}).get("symbol") or "").strip().upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    symbols.append(sym)
                if len(symbols) >= max_symbols:
                    break

            summary["input"] = int(len(symbols))
            if not symbols:
                status = "SKIP"
            else:
                from .research.candidate_scanner import CandidateScanner

                scanner = CandidateScanner(self.db, self.config, log=self._logger)
                out = []
                src_scan_id = str((payload or {}).get("scan_id") or "")
                for sym in symbols:
                    row = scanner.score_single_symbol(
                        sym,
                        universe="SIMULATION",
                        extra_details={"source": "AGENT_SIMULATION", "source_scan_id": src_scan_id},
                        force_accept=False,
                    )
                    if row:
                        out.append(row)

                sim_scan_id = f"SIM_{datetime.now(timezone.utc).strftime('%Y.%m.%d_%H.%M.%S')}"
                summary["scan_id"] = sim_scan_id
                summary["scored"] = int(len(out))
                summary["symbols"] = symbols[:10]
                if out:
                    self.db.save_candidates(sim_scan_id, out, universe="SIMULATION", policy="AGENT_SIMULATION")
                    self.metrics.log_metric("agent_candidate_simulation_rows", float(len(out)), {"scan_id": sim_scan_id, "mode": self.mode})
                    self.publish("candidate_simulation_completed", {"scan_id": sim_scan_id, "count": len(out), "symbols": symbols[:10]})
                    self.log(f"🧠 [AGENT] Candidate simulation complete: {len(out)} row(s) (scan_id={sim_scan_id}).")
                else:
                    status = "SKIP"
                    self.log("🧠 [AGENT] Candidate simulation produced no rows.")
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "CANDIDATE_SIMULATION_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_watchlist_policy_update(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-select-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(run_id, mode=str(self.mode), phase="SELECT", status="OK", summary={"job": "agent_watchlist_policy_update"})
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_watchlist_policy_update"}
        try:
            require_engine = self._cfg_bool("CONFIGURATION", "agent_autopilot_require_engine_running_for_mutations", True)
            if require_engine and not self._engine_running():
                status = "SKIP"
                summary["reason"] = "engine_stopped"
                self.metrics.log_anomaly(
                    "AUTOPILOT_MUTATION_BLOCKED",
                    severity="INFO",
                    source="agent_maintenance",
                    details={"job": "agent_watchlist_policy_update", "reason": "engine_stopped"},
                )
                if run_id:
                    self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
                return

            before = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL") or [])
            with self._state_write_lock:
                res = apply_watchlist_policy(
                self.config,
                self.db,
                getattr(self.db, "paths", {}) or {},
                log=self.log,
                source="agent_master",
                apply_mode=(self._cfg_str("CONFIGURATION", "watchlist_auto_update_mode", "ADD") or "ADD").upper(),
                write_cb=self._persist_split_config,
            )
            added = list(res.get("added") or [])
            removed = list(res.get("removed") or [])
            after = list(res.get("new_watchlist") or get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL") or [])
            rejected = dict(res.get("rejected") or {})

            max_churn = max(0, self._cfg_int("CONFIGURATION", "agent_watchlist_policy_max_churn_per_run", 10))
            churn = len(added) + len(removed)
            if max_churn > 0 and churn > max_churn:
                status = "WARN"
                self.metrics.log_anomaly(
                    "WATCHLIST_POLICY_CHURN_EXCEEDED",
                    severity="WARN",
                    source="agent_maintenance",
                    details={"churn": churn, "max_churn": max_churn},
                )

            self.publish("watchlist_policy_updated", {"added": added, "removed": removed, "rejected": rejected, "before": before, "after": after})
            summary.update({"added": added, "removed": removed, "rejected_count": len(rejected), "before_count": len(before), "after_count": len(after), "batch_id": res.get("batch_id")})
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "WATCHLIST_POLICY_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_quick_backtests(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-backtest-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(run_id, mode=str(self.mode), phase="BACKTEST", status="OK", summary={"job": "agent_quick_backtests"})
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_quick_backtests", "symbols": 0, "ok": 0}
        try:
            max_symbols = max(1, self._cfg_int("CONFIGURATION", "agent_quick_backtest_max_symbols", 10))
            bt_days = max(1, self._cfg_int("CONFIGURATION", "agent_quick_backtest_days", 14))
            max_strats = max(1, self._cfg_int("CONFIGURATION", "agent_quick_backtest_max_strategies", 6))
            min_trades = max(0, self._cfg_int("CONFIGURATION", "agent_quick_backtest_min_trades", 1))

            latest = self.db.get_latest_candidates(limit=max_symbols) if hasattr(self.db, "get_latest_candidates") else []
            rows = []
            try:
                if hasattr(latest, "to_dict"):
                    rows = list(latest.to_dict("records"))
                elif isinstance(latest, list):
                    rows = list(latest)
            except Exception:
                rows = []

            symbols = []
            seen = set()
            for r in rows:
                sym = str((r or {}).get("symbol") or "").strip().upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    symbols.append(sym)
                if len(symbols) >= max_symbols:
                    break

            summary["symbols"] = len(symbols)
            if not symbols:
                status = "SKIP"
            else:
                from .research.quick_backtest import run_quick_backtest

                ok_count = 0
                for sym in symbols:
                    res = run_quick_backtest(
                        self.config,
                        self.db,
                        sym,
                        days=bt_days,
                        max_strategies=max_strats,
                        min_trades=min_trades,
                    )
                    best = (res or {}).get("best") or {}
                    out = {
                        "symbol": sym,
                        "strategy": str(best.get("strategy") or "QUICK_BACKTEST"),
                        "start_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "end_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "win_rate": float(best.get("win_rate") or 0.0),
                        "total_profit": float(best.get("total_pl") or 0.0),
                        "max_drawdown": float(best.get("max_drawdown") or 0.0),
                        "sharpe_ratio": 0.0,
                        "trade_count": int(best.get("trades") or 0),
                        "best_params": "",
                        "tested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "results_json": json.dumps(res or {}),
                    }
                    self.db.save_backtest_result(out)
                    if bool((res or {}).get("ok")):
                        ok_count += 1

                summary["ok"] = ok_count
                self.metrics.log_metric("agent_quick_backtest_symbols", float(len(symbols)), {"ok": ok_count, "mode": self.mode})
                self.publish("quick_backtests_completed", {"count": len(symbols), "ok": ok_count, "symbols": symbols[:10]})
                self.log(f"🧠 [AGENT] Quick backtests complete: {ok_count}/{len(symbols)} ok.")
                if ok_count == 0:
                    status = "WARN"
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "QUICK_BACKTEST_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_architect_optimize(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-architect-opt-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(run_id, mode=str(self.mode), phase="SELECT", status="OK", summary={"job": "agent_architect_optimize"})
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_architect_optimize", "symbols": 0, "queued": 0}
        try:
            max_symbols = max(1, self._cfg_int("CONFIGURATION", "agent_architect_optimize_max_symbols", 5))
            top_k = max(1, self._cfg_int("CONFIGURATION", "agent_architect_optimize_top_variants_per_symbol", 3))

            latest = self.db.get_latest_candidates(limit=max_symbols) if hasattr(self.db, "get_latest_candidates") else []
            rows = []
            try:
                if hasattr(latest, "to_dict"):
                    rows = list(latest.to_dict("records"))
                elif isinstance(latest, list):
                    rows = list(latest)
            except Exception:
                rows = []

            symbols = []
            seen = set()
            for r in rows:
                sym = str((r or {}).get("symbol") or "").strip().upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    symbols.append(sym)
                if len(symbols) >= max_symbols:
                    break

            summary["symbols"] = len(symbols)
            if not symbols:
                status = "SKIP"
            elif not hasattr(self.db, "architect_queue_enqueue"):
                status = "WARN"
                summary["warning"] = "architect_queue_enqueue_unavailable"
            else:
                from .architect import TheArchitect

                optimizer = TheArchitect(self.db, config=self.config)
                queued = 0
                for sym in symbols:
                    try:
                        variants = optimizer.run_optimization(sym, lambda *_a, **_k: None) or []
                    except Exception:
                        variants = []
                    if not variants:
                        continue

                    variants = sorted(
                        variants,
                        key=lambda v: float((v or {}).get("score", (v or {}).get("profit", 0.0)) or 0.0),
                        reverse=True,
                    )

                    local_seen = set()
                    for variant in variants:
                        genome = {
                            "rsi": (variant or {}).get("rsi"),
                            "sl": (variant or {}).get("sl"),
                            "tp": (variant or {}).get("tp"),
                            "ema": bool((variant or {}).get("ema")),
                        }
                        gkey = (genome["rsi"], genome["sl"], genome["tp"], genome["ema"])
                        if gkey in local_seen:
                            continue
                        local_seen.add(gkey)

                        metrics = {
                            "profit": (variant or {}).get("profit"),
                            "trades": (variant or {}).get("trades"),
                            "win_rate": (variant or {}).get("win_rate"),
                            "score": (variant or {}).get("score"),
                        }
                        item_id = self.db.architect_queue_enqueue(sym, genome, metrics=metrics)
                        if item_id:
                            queued += 1
                        if len(local_seen) >= top_k:
                            break

                summary["queued"] = queued
                self.metrics.log_metric("agent_architect_optimize_queued", float(queued), {"symbols": len(symbols), "mode": self.mode})
                self.publish("architect_optimize_completed", {"symbols": symbols[:10], "queued": queued})
                self.log(f"🧠 [AGENT] Architect optimize complete: queued {queued} variants across {len(symbols)} symbols.")
                if queued == 0:
                    status = "WARN"
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "ARCHITECT_OPTIMIZE_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_architect_orchestrator(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-architect-orch-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(run_id, mode=str(self.mode), phase="DEPLOY", status="OK", summary={"job": "agent_architect_orchestrator"})
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_architect_orchestrator", "queue_items": 0, "symbols": 0, "processed": 0}
        try:
            target_hour = self._cfg_int("CONFIGURATION", "agent_architect_orchestrator_hour_utc", 3)
            now = datetime.now(timezone.utc)
            if int(now.hour) != int(target_hour):
                status = "SKIP"
                summary["reason"] = "off_hours_gate"
            elif not hasattr(self.db, "architect_queue_list"):
                status = "WARN"
                summary["warning"] = "architect_queue_list_unavailable"
            else:
                max_queue = max(1, self._cfg_int("CONFIGURATION", "agent_architect_orchestrator_max_queue_items", 10))
                max_symbols = max(1, self._cfg_int("CONFIGURATION", "agent_architect_orchestrator_max_symbols", 25))

                queue_items = self.db.architect_queue_list(statuses=["NEW"], limit=max_queue)
                summary["queue_items"] = len(queue_items)
                if not queue_items:
                    status = "SKIP"
                    summary["reason"] = "queue_empty"
                else:
                    from .backtest_runner import normalize_variants, run_architect_queue_backtest
                    from .research.architect_backtest_exporter import export_architect_backtest_bundle
                    from .utils import APP_VERSION, APP_RELEASE

                    variants = normalize_variants(queue_items)
                    for item in queue_items:
                        try:
                            if hasattr(self.db, "architect_queue_mark_status"):
                                self.db.architect_queue_mark_status(item.get("id"), "RUNNING")
                        except Exception:
                            pass

                    symbols = []
                    try:
                        symbols = get_watchlist_symbols(self.config, group="ACTIVE") or []
                    except Exception:
                        symbols = []
                    if not symbols and hasattr(self.db, "get_latest_candidates"):
                        try:
                            cdf = self.db.get_latest_candidates(limit=max_symbols)
                            rows = cdf.to_dict("records") if hasattr(cdf, "to_dict") else list(cdf or [])
                            for r in rows:
                                s = str((r or {}).get("symbol") or "").strip().upper()
                                if s and s not in symbols:
                                    symbols.append(s)
                                if len(symbols) >= max_symbols:
                                    break
                        except Exception:
                            symbols = []
                    symbols = symbols[:max_symbols]
                    summary["symbols"] = len(symbols)

                    if not symbols:
                        status = "WARN"
                        summary["warning"] = "no_symbols"
                    else:
                        result = run_architect_queue_backtest(
                            db_manager=self.db,
                            config=self.config,
                            variants=variants,
                            symbols=symbols,
                            history_limit=3000,
                            max_workers=4,
                            progress_cb=None,
                            score_key='score',
                        )
                        logs_root = getattr(self.db, "paths", {}).get("logs") if getattr(self.db, "paths", None) else None
                        if not logs_root:
                            logs_root = os.path.join(os.getcwd(), "logs")
                        out = export_architect_backtest_bundle(
                            result=result,
                            config=self.config,
                            out_dir=os.path.join(logs_root, "backtest"),
                            summaries_dir=os.path.join(logs_root, "summaries"),
                            app_version=APP_VERSION,
                            app_release=APP_RELEASE,
                            include_csv=True,
                        )
                        bundle_json = ""
                        try:
                            bundle_json = str((out or {}).get("json") or "")
                        except Exception:
                            bundle_json = ""

                        processed = 0
                        for item in queue_items:
                            ok = False
                            try:
                                if hasattr(self.db, "architect_queue_mark_status"):
                                    ok = bool(self.db.architect_queue_mark_status(item.get("id"), "DONE", result_pointer=bundle_json))
                            except Exception:
                                ok = False
                            if ok:
                                processed += 1
                        summary["processed"] = processed
                        self.metrics.log_metric("agent_architect_orchestrator_processed", float(processed), {"symbols": len(symbols), "mode": self.mode})
                        self.publish("architect_orchestrator_completed", {"processed": processed, "symbols": symbols[:10], "bundle": bundle_json})

                        try:
                            dep_id = f"DEPLOY-{int(time.time())}-ARCH-ORCH"
                            self.experiments.create_deployment_unit(
                                deployment_id=dep_id,
                                change_type="ARCHITECT_ORCHESTRATOR_BUNDLE",
                                diff_text=f"queue_items={len(queue_items)}; symbols={len(symbols)}; processed={processed}",
                                status=("APPLIED" if processed > 0 else "PROPOSED"),
                                approved_by="agent_master",
                                rollback_pointer=bundle_json,
                            )
                        except Exception:
                            try:
                                self._logger.exception("[E_DEPLOYMENT_UNIT_ARCH_ORCH] Failed to write deployment unit")
                            except Exception:
                                pass

                        self.log(f"🧠 [AGENT] Architect orchestrator complete: processed {processed} queue items across {len(symbols)} symbols.")
                        if processed == 0:
                            status = "WARN"
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "ARCHITECT_ORCHESTRATOR_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

    def _run_full_backtests(self) -> None:
        if self.mode not in {"PAPER", "LIVE"}:
            return

        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day == self._last_full_backtest_date:
            return

        target_hour = self._cfg_int("CONFIGURATION", "agent_full_backtest_hour_utc", 2)
        if int(now.hour) != int(target_hour):
            return

        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-full-backtest-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(run_id, mode=str(self.mode), phase="BACKTEST", status="OK", summary={"job": "agent_full_backtests"})
        except Exception:
            run_id = ""

        status = "OK"
        summary = {"job": "agent_full_backtests"}
        try:
            service_res = run_full_backtest_service(
                self.config,
                self.db,
                simulate_strategy=None,
                log=self.log,
                rebuild_table=True,
                sleep_per_symbol_sec=0.0,
            )
            summary.update(service_res if isinstance(service_res, dict) else {})
            if not bool((service_res or {}).get("ok")):
                status = "WARN"
            self._last_full_backtest_date = day
        except Exception as e:
            status = "FAIL"
            summary["error"] = str(e)
            self.metrics.log_anomaly(
                "FULL_BACKTEST_FAIL",
                severity="WARN",
                source="agent_maintenance",
                details={"error": str(e)},
            )

        if run_id:
            try:
                self.metrics.finish_autopilot_run(run_id, status=status, summary=summary)
            except Exception:
                pass

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
                with self._state_write_lock:
                    write_split_config(self.config, p)
        except Exception:
            self._logger.exception("[E_AGENT_CFG_WRITE] Failed persisting split config")

    def _health_snapshot(self):
        run_id = ""
        try:
            self._autopilot_seq = int(getattr(self, "_autopilot_seq", 0) or 0) + 1
            run_id = f"{int(time.time())}-observe-{self._autopilot_seq}"
            self.metrics.start_autopilot_run(
                run_id,
                mode=str(self.mode),
                phase="OBSERVE",
                status="OK",
                summary={"job": "agent_health_snapshot"},
            )
        except Exception:
            run_id = ""

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

        if run_id:
            try:
                self.metrics.finish_autopilot_run(
                    run_id,
                    status="OK",
                    summary={
                        "job": "agent_health_snapshot",
                        "mode": self.mode,
                        "freshness_seconds": freshness,
                        "api_error_streak": api_err,
                        "reject_ratio": reject_ratio,
                    },
                )
            except Exception:
                pass

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

    def _cfg_float(self, section: str, key: str, default: float = 0.0) -> float:
        try:
            return float(str(self.config.get(section, key, fallback=str(default))).strip())
        except Exception:
            return float(default)

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

        if ev_type == "candidate_scan_completed":
            if self._cfg_bool("CONFIGURATION", "agent_candidate_simulation_run_after_scan", True):
                self._run_candidate_simulation(payload)
            return

        if ev_type == "candidate_simulation_completed":
            if self._cfg_bool("CONFIGURATION", "agent_watchlist_policy_run_after_simulation", True):
                self._run_watchlist_policy_update()
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
