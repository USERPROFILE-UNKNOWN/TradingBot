
import os
import glob
import json
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
            self._scan_summaries(os.path.join(logs_dir, "summaries"))

        if self._get_cfg_bool("agent_shadow_include_backtests", True):
            # v5.13.0 updateB: support both legacy backtests/ and current backtest/ folders
            self._scan_backtests(os.path.join(logs_dir, "backtest"))
            self._scan_backtests(os.path.join(logs_dir, "backtests"))

    def _scan_summaries(self, summaries_dir: str):
        if not os.path.isdir(summaries_dir):
            return

        files = sorted(glob.glob(os.path.join(summaries_dir, "summary_*.json")), key=os.path.getmtime, reverse=True)
        for path in files[:5]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            df = data.get("data_freshness") or {}
            stale_count = int(df.get("stale_count") or 0)
            stale_symbols = df.get("stale_symbols") or []

            if stale_count > 0:
                title = f"Data freshness: {stale_count} stale symbols"
                payload = {
                    "kind": "action",
                    "action": "BACKFILL_DB",
                    "symbols": stale_symbols,
                    "source": "run_summary",
                }
                sid = self.db.upsert_agent_suggestion(
                    artifact_type="summary",
                    artifact_path=path,
                    title=title,
                    suggestion_type="action",
                    suggestion_payload=payload,
                    status="NEW",
                )
                if sid:
                    self.db.add_agent_rationale(
                        sid,
                        rationale="Run Summary indicates stale symbols (data is older than freshness window). Consider running a DB backfill or increasing lookback.",
                        metrics_payload={"stale_count": stale_count},
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

                payload = {
                    "kind": "config_change",
                    "config_changes": {lookback_key: str(new_val)} if new_val else {},
                    "source": "run_summary",
                }
                sid = self.db.upsert_agent_suggestion(
                    artifact_type="summary",
                    artifact_path=path,
                    title=title,
                    suggestion_type="config_change",
                    suggestion_payload=payload,
                    status="NEW",
                )
                if sid:
                    self.db.add_agent_rationale(
                        sid,
                        rationale="Run Summary shows AI is not trained or ROC-AUC is low. More historical data can improve training stability.",
                        metrics_payload={"roc_auc": roc_auc, "model_status": model_status},
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
                        payload = {
                            "kind": "note",
                            "source": "architect_backtest_bundle",
                            "variant": vid,
                            "metrics": {
                                "profit_sum": ps,
                                "symbols": m.get("symbols"),
                                "positive_symbols": (m.get("symbols_positive") if isinstance(m, dict) else None) or m.get("positive_symbols"),
                                "profit_mean": m.get("profit_mean"),
                            },
                        }
                        sid = self.db.upsert_agent_suggestion(
                            artifact_type="backtest",
                            artifact_path=path,
                            title=title,
                            suggestion_type="note",
                            suggestion_payload=payload,
                            status="NEW",
                        )
                        if sid:
                            self.db.add_agent_rationale(
                                sid,
                                rationale="An orchestrated Architect run identifies a variant with the best aggregate profitability across the selected universe.",
                                metrics_payload=payload.get("metrics") or {},
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
                payload = {
                    "kind": "note",
                    "source": "backtest_export",
                    "symbol": sym,
                    "metrics": {"trades": trades, "win_rate": win_rate},
                }
                sid = self.db.upsert_agent_suggestion(
                    artifact_type="backtest",
                    artifact_path=path,
                    title=title,
                    suggestion_type="note",
                    suggestion_payload=payload,
                    status="NEW",
                )
                if sid:
                    self.db.add_agent_rationale(
                        sid,
                        rationale="Recent backtest export shows a low win-rate with sufficient trades. Consider revisiting strategy parameters for this symbol.",
                        metrics_payload={"symbol": sym, "trades": trades, "win_rate": win_rate},
                    )

