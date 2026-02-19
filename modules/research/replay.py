"""v5.12.6 updateA - Replay harness + decision/execution packets.

This module is intentionally standalone and safe to run from a frozen EXE.
It reads packet tables from decision_logs.db and produces a diff-style report.

Usage (from repo root):
    python -m TradingBot.modules.research.replay --symbol AAPL

Or:
    python TradingBot\modules\research\replay.py --symbol AAPL
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime

from ..paths import get_paths


def _utc_now_stamp() -> str:
    try:
        return datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S")
    except Exception:
        return "replay"


def _connect(db_path: str) -> sqlite3.Connection:
    # isolation_level=None -> autocommit reads; check_same_thread=False allows UI threads if needed
    return sqlite3.connect(db_path, check_same_thread=False)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def replay(
    *,
    tradingbot_root: str | None = None,
    decision_db_path: str | None = None,
    symbol: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    output_path: str | None = None,
    max_rows: int = 50000,
) -> dict:
    """Replay a recent window and return a structured diff report.

    start_ts/end_ts are optional and interpreted as ISO strings (SQLite DATETIME text).
    If omitted, returns the most recent max_rows worth of packets.
    """

    if not decision_db_path:
        prev_root = os.environ.get("TRADINGBOT_ROOT")
        if tradingbot_root:
            os.environ["TRADINGBOT_ROOT"] = tradingbot_root
        try:
            paths = get_paths()
        finally:
            if prev_root is None:
                os.environ.pop("TRADINGBOT_ROOT", None)
            else:
                os.environ["TRADINGBOT_ROOT"] = prev_root
        decision_db_path = os.path.join(paths["db"], "decision_logs.db")

    report = {
        "meta": {
            "decision_db_path": decision_db_path,
            "symbol": symbol,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "generated_utc": _utc_now_stamp(),
        },
        "counts": {},
        "mismatches": [],
        "examples": {},
    }

    if not os.path.exists(decision_db_path):
        report["error"] = f"decision_logs.db not found at: {decision_db_path}"
        return report

    conn = _connect(decision_db_path)
    try:
        cur = conn.cursor()

        # --- Decisions
        where = []
        params = []
        if symbol:
            where.append("symbol = ?")
            params.append(symbol.upper())
        if start_ts:
            where.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            where.append("timestamp <= ?")
            params.append(end_ts)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        q_dec = (
            "SELECT decision_id, timestamp, symbol, strategy, action, score, price, ai_prob, sentiment, reason, market_regime, is_crypto, payload_json "
            f"FROM decision_packets {where_sql} ORDER BY timestamp DESC LIMIT ?"
        )
        cur.execute(q_dec, params + [max_rows])
        decisions = cur.fetchall() or []

        # --- Executions
        q_exe = (
            "SELECT event_id, timestamp, decision_id, symbol, side, phase, qty, price, order_id, client_order_id, broker_status, payload_json "
            f"FROM execution_packets {where_sql} ORDER BY timestamp DESC LIMIT ?"
        )
        cur.execute(q_exe, params + [max_rows])
        execs = cur.fetchall() or []

    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Index
    decisions_by_id = {}
    buy_decisions = []
    for row in decisions:
        (did, ts, sym, strat, action, score, price, ai_prob, sent, reason, regime, is_crypto, payload_json) = row
        d = {
            "decision_id": did,
            "timestamp": ts,
            "symbol": sym,
            "strategy": strat,
            "action": action,
            "score": _safe_float(score),
            "price": _safe_float(price),
            "ai_prob": _safe_float(ai_prob),
            "sentiment": _safe_float(sent),
            "reason": reason,
            "market_regime": regime,
            "is_crypto": bool(is_crypto),
        }
        try:
            d["payload"] = json.loads(payload_json) if payload_json else {}
        except Exception:
            d["payload"] = {}
        decisions_by_id[did] = d
        if str(action).upper() == "BUY":
            buy_decisions.append(did)

    execs_by_decision = {}
    fills_without_decision = []
    submits_per_client = {}

    for row in execs:
        (eid, ts, did, sym, side, phase, qty, price, oid, coid, status, payload_json) = row
        e = {
            "event_id": eid,
            "timestamp": ts,
            "decision_id": did,
            "symbol": sym,
            "side": side,
            "phase": phase,
            "qty": _safe_float(qty),
            "price": _safe_float(price),
            "order_id": oid,
            "client_order_id": coid,
            "broker_status": status,
        }
        try:
            e["payload"] = json.loads(payload_json) if payload_json else {}
        except Exception:
            e["payload"] = {}

        if did:
            execs_by_decision.setdefault(did, []).append(e)
        else:
            if str(phase).upper() == "FILL":
                fills_without_decision.append(e)

        if str(phase).upper() == "SUBMIT" and coid:
            submits_per_client[coid] = submits_per_client.get(coid, 0) + 1

    # Mismatch checks
    missing_exec_for_buy = []
    missing_fill_for_buy = []

    for did in buy_decisions:
        events = execs_by_decision.get(did, [])
        phases = {str(ev.get("phase", "")).upper() for ev in events}
        if not phases.intersection({"INTENT", "SUBMIT", "ADOPT"}):
            missing_exec_for_buy.append(did)
        if "FILL" not in phases:
            missing_fill_for_buy.append(did)

    dup_submits = {k: v for k, v in submits_per_client.items() if v and v > 1}

    report["counts"] = {
        "decision_packets": len(decisions),
        "buy_decisions": len(buy_decisions),
        "execution_events": len(execs),
        "buy_missing_exec": len(missing_exec_for_buy),
        "buy_missing_fill": len(missing_fill_for_buy),
        "fills_without_decision": len(fills_without_decision),
        "duplicate_submits_by_client_order_id": len(dup_submits),
    }

    # Build mismatch list with a few concrete examples
    def add(kind: str, items, limit=25):
        if not items:
            return
        if isinstance(items, dict):
            ex = list(items.items())[:limit]
        else:
            ex = items[:limit]
        report["mismatches"].append({"type": kind, "count": len(items), "examples": ex})

    add("BUY_DECISION_WITHOUT_EXEC", missing_exec_for_buy)
    add("BUY_DECISION_WITHOUT_FILL", missing_fill_for_buy)
    add("FILL_WITHOUT_DECISION", fills_without_decision)
    add("DUPLICATE_SUBMITS_SAME_CLIENT_ID", dup_submits)

    # Attach small examples for quick inspection
    try:
        if buy_decisions:
            did0 = buy_decisions[0]
            report["examples"]["latest_buy_decision"] = decisions_by_id.get(did0)
            report["examples"]["latest_buy_exec_events"] = execs_by_decision.get(did0, [])[:25]
    except Exception:
        pass

    # Output file
    if not output_path:
        try:
            prev_root = os.environ.get("TRADINGBOT_ROOT")
            if tradingbot_root:
                os.environ["TRADINGBOT_ROOT"] = tradingbot_root
            try:
                paths = get_paths()
            finally:
                if prev_root is None:
                    os.environ.pop("TRADINGBOT_ROOT", None)
                else:
                    os.environ["TRADINGBOT_ROOT"] = prev_root
            out_dir = os.path.join(paths["logs"], "research")
        except Exception:
            out_dir = os.path.join(os.path.dirname(decision_db_path), "..", "logs", "research")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"replay_diff_{_utc_now_stamp()}.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["output_path"] = output_path
    except Exception as e:
        report["output_error"] = f"{type(e).__name__}: {e}"

    return report


def main():
    p = argparse.ArgumentParser(description="TradingBot replay harness (decision/execution packets)")
    p.add_argument("--root", default=None, help="TradingBot root folder (optional)")
    p.add_argument("--db", default=None, help="Path to decision_logs.db (optional)")
    p.add_argument("--symbol", default=None, help="Filter to a single symbol")
    p.add_argument("--start", default=None, help="Start timestamp (SQLite text/ISO)")
    p.add_argument("--end", default=None, help="End timestamp (SQLite text/ISO)")
    p.add_argument("--out", default=None, help="Write report to a specific JSON path")
    args = p.parse_args()

    rep = replay(
        tradingbot_root=args.root,
        decision_db_path=args.db,
        symbol=args.symbol,
        start_ts=args.start,
        end_ts=args.end,
        output_path=args.out,
    )

    # Minimal console summary
    counts = rep.get("counts", {}) or {}
    print("REPLAY REPORT")
    for k in sorted(counts.keys()):
        print(f"- {k}: {counts[k]}")
    if rep.get("output_path"):
        print(f"\nWrote: {rep['output_path']}")


if __name__ == "__main__":
    main()
