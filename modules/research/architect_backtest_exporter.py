"""Architect backtest (orchestrator) export utilities.

v5.13.0 updateB_architect_backtest_orchestration

Creates a stable artifact bundle in logs/backtest/ and a summary record in
logs/summaries/.

Bundle includes:
- Variant genomes (parameters)
- Per-symbol results (wide format)
- Aggregates (per-variant + best counts)

Important:
- Does NOT export secrets (skips the KEYS section).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_get_section(config: Any, section: str) -> Dict[str, str]:
    try:
        if config is None:
            return {}
        if hasattr(config, "has_section") and config.has_section(section):
            return {k: str(v) for k, v in config[section].items()}
    except Exception:
        pass
    return {}


def _extract_knobs(config: Any) -> Dict[str, str]:
    cfg = _safe_get_section(config, "CONFIGURATION")
    out: Dict[str, str] = {}

    for k, v in cfg.items():
        lk = str(k).lower()
        if lk.startswith("backtest_") or lk.startswith("architect_"):
            out[str(k)] = str(v)

    return out


def _params_hash(payload: Dict[str, Any]) -> str:
    try:
        b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        b = repr(payload).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def export_architect_backtest_bundle(
    *,
    orchestrator_result: Dict[str, Any],
    config: Any,
    out_dir: str,
    summaries_dir: Optional[str] = None,
    app_version: str = "",
    app_release: str = "",
    include_csv: bool = True,
) -> Optional[Dict[str, str]]:
    """Export an Architect Orchestrator bundle.

    orchestrator_result: output of modules.backtest_runner.run_architect_queue_backtest()

    Returns dict with keys: json, csv (optional), summary.
    """

    if not orchestrator_result:
        return None

    df = orchestrator_result.get("df")
    try:
        if df is None or getattr(df, "empty", False):
            return None
    except Exception:
        return None

    os.makedirs(out_dir, exist_ok=True)

    if not summaries_dir:
        # best-effort: sibling to out_dir
        try:
            root = os.path.dirname(out_dir.rstrip("\\/"))
            summaries_dir = os.path.join(root, "summaries")
        except Exception:
            summaries_dir = None

    if summaries_dir:
        try:
            os.makedirs(summaries_dir, exist_ok=True)
        except Exception:
            summaries_dir = None

    ts_local = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    base = f"architect_backtest_bundle_{ts_local}"

    json_path = os.path.join(out_dir, f"{base}.json")
    csv_path = os.path.join(out_dir, f"{base}_results.csv")

    # ---- Snapshot rows (stable) ----
    try:
        cols = list(getattr(df, "columns", []))
        rows = []
        for _, r in df.iterrows():
            d = {}
            for c in cols:
                v = r[c]
                try:
                    if hasattr(v, "item"):
                        v = v.item()
                except Exception:
                    pass
                d[str(c)] = v
            rows.append(d)
    except Exception:
        return None

    variants = orchestrator_result.get("variants") or []
    aggregates = orchestrator_result.get("aggregates") or {}
    meta = orchestrator_result.get("meta") or {}

    params_payload = {
        "knobs": _extract_knobs(config),
        "variants": variants,
    }
    params_hash = _params_hash(params_payload)

    bundle = {
        "schema": {"name": "TradingBot.architect_backtest_bundle", "version": 1},
        "exported_at_utc": _utc_iso_now(),
        "app": {"version": app_version, "release": app_release},
        "meta": meta,
        "params": params_payload,
        "params_hash": params_hash,
        "results": {"rows": rows, "row_count": len(rows)},
        "aggregates": aggregates,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    out = {"json": json_path}

    if include_csv:
        try:
            # Prefer pandas native exporter if available.
            if hasattr(df, "to_csv"):
                df.to_csv(csv_path, index=False)
            else:
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    for row in rows:
                        w.writerow({k: row.get(k) for k in cols})
            out["csv"] = csv_path
        except Exception:
            pass

    # ---- Summary record (best-effort) ----
    if summaries_dir:
        try:
            # Top variants by profit_sum if present
            top_variants = []
            per_variant = (aggregates.get("per_variant") or {}) if isinstance(aggregates, dict) else {}
            for vid, a in (per_variant or {}).items():
                try:
                    top_variants.append({
                        "variant": vid,
                        "profit_sum": a.get("profit_sum"),
                        "profit_mean": a.get("profit_mean"),
                        "symbols": a.get("symbols"),
                        "symbols_positive": a.get("symbols_positive"),
                        "symbol_win_rate": a.get("symbol_win_rate"),
                        "trades_sum": a.get("trades_sum"),
                    })
                except Exception:
                    continue

            def _k(x):
                try:
                    return float(x.get("profit_sum") or 0.0)
                except Exception:
                    return 0.0

            top_variants.sort(key=_k, reverse=True)
            top_variants = top_variants[:10]

            # Top symbols by best_profit
            top_symbols = []
            for row in rows:
                try:
                    top_symbols.append({
                        "symbol": row.get("symbol"),
                        "best_variant": row.get("best_variant"),
                        "best_profit": row.get("best_profit"),
                        "best_score": row.get("best_score"),
                        "best_win_rate": row.get("best_win_rate"),
                        "best_trades": row.get("best_trades"),
                    })
                except Exception:
                    continue

            def _ks(x):
                try:
                    return float(x.get("best_profit") or 0.0)
                except Exception:
                    return 0.0

            top_symbols.sort(key=_ks, reverse=True)
            top_symbols = top_symbols[:25]

            summary = {
                "schema": {"name": "TradingBot.architect_backtest_summary", "version": 1},
                "exported_at_utc": _utc_iso_now(),
                "bundle_path": json_path,
                "top_variants": top_variants,
                "top_symbols": top_symbols,
            }

            sname = f"summary_{ts_local}_architect_backtest.json"
            spath = os.path.join(summaries_dir, sname)
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            out["summary"] = spath
        except Exception:
            pass

    return out
