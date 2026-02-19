from modules.metrics import MetricsStore


def test_metrics_store_v5170_health_tables_and_snapshot(tmp_path):
    store = MetricsStore(str(tmp_path))

    store.log_job_run("agent_health_snapshot", "ok", 123, {"note": "tick"})
    store.log_anomaly("DATA_GAP", severity="WARN", source="test", details={"symbol": "AAPL"})
    store.log_symbol_health(
        "SYSTEM",
        freshness_seconds=95.0,
        api_error_streak=2,
        reject_ratio=0.10,
        decision_exec_latency_ms=250.0,
        slippage_bps=1.2,
        details={"mode": "PAPER"},
    )

    snap = store.get_health_widget_snapshot(lookback_sec=3600)
    assert snap["data_freshness_seconds"] == 95.0
    assert snap["api_error_streak"] == 2
    assert snap["order_reject_ratio"] == 0.10
    assert snap["decision_exec_latency_ms"] == 250.0
    assert snap["slippage_bps"] == 1.2
    assert snap["anomaly_count"] >= 1


def test_metrics_store_v620_autopilot_and_job_state(tmp_path):
    store = MetricsStore(str(tmp_path))

    store.update_job_state("agent_health_snapshot", "ok")
    st = store.get_job_state("agent_health_snapshot")
    assert st.get("last_attempt_at") is not None
    assert st.get("last_success_at") is not None
    assert st.get("last_error") == ""

    store.update_job_state("agent_health_snapshot", "error", error="boom", cooldown_until=999)
    st2 = store.get_job_state("agent_health_snapshot")
    assert st2.get("last_attempt_at") is not None
    assert st2.get("last_error") == "boom"
    assert st2.get("cooldown_until") == 999

    store.start_autopilot_run("r1", mode="PAPER", phase="SCAN", status="OK", summary={"a": 1})
    store.finish_autopilot_run("r1", status="WARN", summary={"b": 2})

    with store._connect() as conn:
        row = conn.execute(
            "SELECT mode, phase, status, ended_at FROM autopilot_runs WHERE run_id = ?",
            ("r1",),
        ).fetchone()
    assert row is not None
    assert row[0] == "PAPER"
    assert row[1] == "SCAN"
    assert row[2] == "WARN"
    assert row[3] is not None


def test_update_job_state_preserves_last_success_on_error(tmp_path):
    from modules.metrics import MetricsStore

    m = MetricsStore(str(tmp_path))
    m.update_job_state("demo", "ok")
    st1 = m.get_job_state("demo")
    assert st1.get("last_success_at") is not None

    m.update_job_state("demo", "error", error="boom")
    st2 = m.get_job_state("demo")
    assert st2.get("last_success_at") == st1.get("last_success_at")
    assert st2.get("last_error") == "boom"
