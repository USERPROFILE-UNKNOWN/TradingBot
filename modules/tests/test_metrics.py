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
