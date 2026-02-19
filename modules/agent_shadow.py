
import os
import glob
import json
import hashlib
import time
from datetime import datetime, timedelta


class AgentShadow:
    # Shadow mode agent: reads artifacts + emits suggestions to DB; never trades or mutates runtime state.

    def __init__(self, config, db_manager, paths: dict, log_fn=None):
        self.config = config
        self.db = db_manager
        self.paths = paths or {}
        self.log = log_fn or (lambda *a, **k: None)
        self._stop = False
        self._thread = None
        self._last_scan = 0.0

    def start(self):
        import threading
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="AgentShadow", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True

    def _get_cfg_bool(self, key: str, default: bool) -> bool:
        try:
            v = self.config.get("CONFIGURATION", key, fallback=str(default))
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            return default

    def _get_cfg_int(self, key: str, default: int) -> int:
        try:
            v = self.config.get("CONFIGURATION", key, fallback=str(default))
            return int(str(v).strip())
        except Exception:
            return default

    def _required_approval_level(self) -> str:
        try:
            mode = str(self.config.get("CONFIGURATION", "agent_mode", fallback="OFF")).strip().upper()
        except Exception:
            mode = "OFF"
        return "REQUIRE_APPROVAL" if mode == "LIVE" else "AUTO"

    def _emit_proposal(
        self,
        *,
        artifact_type: str,
        artifact_path: str,
        title: str,
        suggestion_type: str,
        target: dict,
        proposed_diff: dict,
        evidence_links: list,
        confidence_score: float,
        risk_impact_estimate: str,
        rationale_text: str,
    ):
        payload = {
            "kind": "proposal",
            "target": dict(target or {}),
            "proposed_diff": dict(proposed_diff or {}),
            "evidence_links": list(evidence_links or []),
            "confidence_score": float(max(0.0, min(1.0, confidence_score))),
            "risk_impact_estimate": str(risk_impact_estimate or "LOW").upper(),
            "required_approval_level": self._required_approval_level(),
            "source": "agent_shadow",
        }
        sid = self.db.upsert_agent_suggestion(
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            title=title,
            suggestion_type=suggestion_type,
            suggestion_payload=payload,
            status="NEW",
        )
        if sid:
            self.db.add_agent_rationale(sid, rationale_text=rationale_text)
        return sid

    def _scan_signature(self, files):
        if not files:
            return "", 0.0, ""
        newest = max(files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0)
        try:
            mtime = float(os.path.getmtime(newest))
        except Exception:
            mtime = 0.0
        raw = f"{newest}|{mtime:.6f}|{len(files)}"
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
        return newest, mtime, digest

    def _should_scan_scope(self, scope: str, files) -> bool:
        latest_path, latest_mtime, latest_hash = self._scan_signature(files)
        if not latest_path:
            return False
        try:
            cp = self.db.get_agent_shadow_checkpoint(scope)
        except Exception:
            cp = None
        if not cp:
            return True
        try:
            if str(cp.get("last_hash") or "") != str(latest_hash or ""):
                return True
            if float(cp.get("last_mtime") or 0.0) < float(latest_mtime):
                return True
        except Exception:
            return True
        return False

    def _update_scope_checkpoint(self, scope: str, files):
        latest_path, latest_mtime, latest_hash = self._scan_signature(files)
        if not latest_path:
            return
        try:
            self.db.upsert_agent_shadow_checkpoint(scope, latest_path, latest_mtime, latest_hash)
        except Exception:
            pass

    def _loop(self):
        while not self._stop:
            enabled = self._get_cfg_bool("agent_shadow_enabled", True)
            interval = self._get_cfg_int("agent_shadow_interval_sec", 300)

            now = time.time()
            if enabled and (now - self._last_scan) >= max(10, interval):
                try:
                    self._scan_once()
                except Exception as e:
                    try:
                        self.log(f"[AGENT] Shadow scan error: {e}")
                    except Exception:
                        pass
                self._last_scan = now

            time.sleep(1.0)

    def _scan_once(self):
        logs_dir = self.paths.get("logs")
        if not logs_dir or not os.path.isdir(logs_dir):
            return

        if self._get_cfg_bool("agent_shadow_include_summaries", True):
            summaries_dir = os.path.join(logs_dir, "summaries")
            summary_files = []
            try:
                summary_files = sorted(glob.glob(os.path.join(summaries_dir, "summary_*.json")), key=os.path.getmtime, reverse=True)
            except Exception:
                summary_files = []
            if self._should_scan_scope("summaries", summary_files):
                self._scan_summaries(summaries_dir)
                self._update_scope_checkpoint("summaries", summary_files)

        if self._get_cfg_bool("agent_shadow_include_backtests", True):
            # v5.13.0 updateB: support both legacy backtests/ and current backtest/ folders
            for scope, folder in (("backtest", os.path.join(logs_dir, "backtest")), ("backtests", os.path.join(logs_dir, "backtests"))):
                files = []
                try:
                    pats = ["architect_backtest_bundle_*.json", "backtest_bundle_*.json", "backtest_*.json"]
                    for pat in pats:
                        files.extend(glob.glob(os.path.join(folder, pat)))
                    files = sorted(set(files), key=os.path.getmtime, reverse=True)
                except Exception:
                    files = []
                if self._should_scan_scope(scope, files):
                    self._scan_backtests(folder)
                    self._update_scope_checkpoint(scope, files)

    def _scan_summaries(self, summaries_dir: str):
        if not os.path.isdir(summaries_dir):
            return

        files = sorted(glob.glob(os.path.join(summaries_dir, "summary_*.json")), key=os.path.getmtime, reverse=True)
        scanned_records = []
        for path in files[:5]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            scanned_records.append({"path": path, "data": data})

            df = data.get("data_freshness") or {}
            stale_count = int(df.get("stale_count") or 0)
            stale_symbols = df.get("stale_symbols") or []

            if stale_count > 0:
                title = f"Data freshness: {stale_count} stale symbols"
                self._emit_proposal(
                    artifact_type="summary",
                    artifact_path=path,
                    title=title,
                    suggestion_type="action",
                    target={"domain": "data_pipeline", "symbols": stale_symbols[:25]},
                    proposed_diff={"action": "BACKFILL_DB", "symbols": stale_symbols},
                    evidence_links=[path],
                    confidence_score=0.93,
                    risk_impact_estimate="LOW",
                    rationale_text="Run Summary indicates stale symbols (data is older than freshness window). Triggering DB backfill should restore freshness.",
                )

            ai = data.get("ai_metrics") or {}
            model_status = str(ai.get("model_status") or "").lower()
            roc_auc = ai.get("roc_auc")
            if model_status in {"not_enough_data", "insufficient_data", "not trained"} or (isinstance(roc_auc, (int, float)) and roc_auc < 0.55):
                title = "AI model quality: consider more training data"
                # Soft suggestion: extend lookback if configured.
                lookback_key = "update_db_lookback_days"
                current = None
                try:
                    current = int(self.db.get_config_value(lookback_key) or 0)
                except Exception:
                    current = None
                new_val = None
                if current is not None and current > 0:
                    new_val = max(current, 60)
                elif current is None:
                    new_val = 60

                before_val = None
                try:
                    before_val = str(current) if current is not None else ""
                except Exception:
                    before_val = ""
                after_val = str(new_val) if new_val is not None else before_val

                self._emit_proposal(
                    artifact_type="summary",
                    artifact_path=path,
                    title=title,
                    suggestion_type="config_change",
                    target={"domain": "config", "key": lookback_key},
                    proposed_diff={"before": {lookback_key: before_val}, "after": {lookback_key: after_val}},
                    evidence_links=[path],
                    confidence_score=0.76,
                    risk_impact_estimate="LOW",
                    rationale_text="Run Summary shows AI is not trained or ROC-AUC is low. Increasing historical lookback can improve training stability.",
                )

        self._emit_repeated_data_quality_proposals(scanned_records)

    def _detect_data_quality_flags(self, data: dict) -> dict:
        flags = {
            "empty_crypto_stable_set": False,
            "mass_quarantine": False,
            "missing_bars_or_nans": False,
        }
        try:
            text = json.dumps(data or {}, ensure_ascii=False).lower()
        except Exception:
            text = ""

        # Empty crypto stable-set selections (best-effort, schema-tolerant)
        if (
            "crypto" in text and "stable" in text and (
                "[]" in text or '"selected_count": 0' in text or "'selected_count': 0" in text or '"stable_set_count": 0' in text
            )
        ):
            flags["empty_crypto_stable_set"] = True

        # Mass quarantine indicators
        try:
            df = (data or {}).get("data_freshness") or {}
            stale_count = int(df.get("stale_count") or 0)
            if stale_count >= 10:
                flags["mass_quarantine"] = True
        except Exception:
            pass
        if "quarantine" in text and ("moved" in text or "archive" in text):
            flags["mass_quarantine"] = True

        # Missing bars / NaNs indicators
        if any(k in text for k in ["missing_bars", "missing bars", "nan_count", "nans", "has_nan", "has_nans"]):
            flags["missing_bars_or_nans"] = True

        return flags

    def _emit_repeated_data_quality_proposals(self, records):
        if not records:
            return
        counts = {
            "empty_crypto_stable_set": 0,
            "mass_quarantine": 0,
            "missing_bars_or_nans": 0,
        }
        evidence = {
            "empty_crypto_stable_set": [],
            "mass_quarantine": [],
            "missing_bars_or_nans": [],
        }

        for rec in records:
            flags = self._detect_data_quality_flags((rec or {}).get("data") or {})
            path = (rec or {}).get("path") or ""
            for k, v in flags.items():
                if v:
                    counts[k] += 1
                    if path:
                        evidence[k].append(path)

        # Repeated threshold: appears in >=2 of latest scanned summaries.
        if counts["empty_crypto_stable_set"] >= 2:
            self._emit_proposal(
                artifact_type="summary",
                artifact_path=(evidence["empty_crypto_stable_set"][0] if evidence["empty_crypto_stable_set"] else ""),
                title="Repeated empty crypto stable-set selections",
                suggestion_type="config_change",
                target={"domain": "candidate_scanner", "segment": "crypto_stable_set"},
                proposed_diff={
                    "before": {"candidate_scanner_max_spread_bps": "current"},
                    "after": {"candidate_scanner_max_spread_bps": "40", "candidate_scanner_lookback_days": "120"},
                },
                evidence_links=evidence["empty_crypto_stable_set"][:5],
                confidence_score=0.72,
                risk_impact_estimate="LOW",
                rationale_text="Repeated empty crypto stable-set selections were observed. Slightly loosening spread/coverage filters may improve candidate formation.",
            )

        if counts["mass_quarantine"] >= 2:
            self._emit_proposal(
                artifact_type="summary",
                artifact_path=(evidence["mass_quarantine"][0] if evidence["mass_quarantine"] else ""),
                title="Repeated mass quarantine / stale-symbol episodes",
                suggestion_type="config_change",
                target={"domain": "stale_quarantine"},
                proposed_diff={
                    "before": {
                        "agent_stale_quarantine_warmup_minutes": "current",
                        "agent_stale_quarantine_threshold_seconds": "current",
                    },
                    "after": {
                        "agent_stale_quarantine_warmup_minutes": "60",
                        "agent_stale_quarantine_threshold_seconds": "28800",
                    },
                },
                evidence_links=evidence["mass_quarantine"][:5],
                confidence_score=0.79,
                risk_impact_estimate="MEDIUM",
                rationale_text="Mass quarantine patterns repeated across summaries. Increasing warm-up and stale threshold can reduce false-positive quarantine churn during fragile data windows.",
            )

        if counts["missing_bars_or_nans"] >= 2:
            self._emit_proposal(
                artifact_type="summary",
                artifact_path=(evidence["missing_bars_or_nans"][0] if evidence["missing_bars_or_nans"] else ""),
                title="Repeated missing-bars / NaN data quality failures",
                suggestion_type="action",
                target={"domain": "data_pipeline", "issue": "missing_bars_or_nans"},
                proposed_diff={"action": "BACKFILL_DB", "note": "trigger_earlier_backfill_and_validate_nans"},
                evidence_links=evidence["missing_bars_or_nans"][:5],
                confidence_score=0.87,
                risk_impact_estimate="LOW",
                rationale_text="Missing bars/NaNs were repeatedly detected. Running earlier backfill and tightening data validation should improve downstream model and scan stability.",
            )

    def _scan_backtests(self, backtests_dir: str):
        """Inspect recent backtest exports and add lightweight suggestions.

        Supported export schemas:
        - TradingBot.architect_backtest_bundle (updateB orchestrator)
        - Legacy backtest_*.json exports with meta/performance keys
        """
        if not os.path.isdir(backtests_dir):
            return

        patterns = [
            "architect_backtest_bundle_*.json",
            "backtest_bundle_*.json",
            "backtest_*.json",
        ]

        files = []
        for pat in patterns:
            try:
                files.extend(glob.glob(os.path.join(backtests_dir, pat)))
            except Exception:
                continue

        files = sorted(set(files), key=os.path.getmtime, reverse=True)
        for path in files[:5]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            schema = None
            try:
                schema = (data.get("schema") or {}).get("name")
            except Exception:
                schema = None

            # --- updateB: orchestrator bundle ---
            if schema == "TradingBot.architect_backtest_bundle":
                try:
                    ag = data.get("aggregates") or {}
                    per = ag.get("per_variant") or {}

                    top = None
                    for vid, m in (per.items() if isinstance(per, dict) else []):
                        try:
                            ps = float(m.get("profit_sum") or m.get("total_profit") or 0.0)
                        except Exception:
                            ps = 0.0
                        if top is None or ps > top[1]:
                            top = (vid, ps, m)

                    if top:
                        vid, ps, m = top
                        title = f"Architect Orchestrator: top variant {vid}"
                        metrics = {
                            "profit_sum": ps,
                            "symbols": m.get("symbols"),
                            "positive_symbols": (m.get("symbols_positive") if isinstance(m, dict) else None) or m.get("positive_symbols"),
                            "profit_mean": m.get("profit_mean"),
                        }
                        self._emit_proposal(
                            artifact_type="backtest",
                            artifact_path=path,
                            title=title,
                            suggestion_type="deploy_strategy",
                            target={"domain": "architect_variant", "variant": vid},
                            proposed_diff={"promote_variant": vid, "metrics": metrics},
                            evidence_links=[path],
                            confidence_score=0.81,
                            risk_impact_estimate="MEDIUM",
                            rationale_text="An orchestrated Architect run identifies a top-performing variant with aggregate profitability across the selected universe.",
                        )
                except Exception:
                    pass
                continue

            # --- ignore backtest bundles here (too high-level for auto notes) ---
            if schema == "TradingBot.backtest_bundle":
                continue

            # --- legacy per-symbol backtest export ---
            meta = data.get("meta") or {}
            sym = meta.get("symbol") or "(unknown)"
            perf = data.get("performance") or {}
            try:
                trades = int(perf.get("trades") or 0)
            except Exception:
                trades = 0
            win_rate = perf.get("win_rate")

            if trades >= 20 and isinstance(win_rate, (int, float)) and win_rate < 0.45:
                title = f"Backtest: low win-rate for {sym}"
                self._emit_proposal(
                    artifact_type="backtest",
                    artifact_path=path,
                    title=title,
                    suggestion_type="config_change",
                    target={"domain": "strategy", "symbol": sym},
                    proposed_diff={"note": "revisit_parameters", "metrics": {"trades": trades, "win_rate": win_rate}},
                    evidence_links=[path],
                    confidence_score=0.68,
                    risk_impact_estimate="MEDIUM",
                    rationale_text="Recent backtest export shows a low win-rate with sufficient trades. Revisit strategy parameters for this symbol.",
                )

