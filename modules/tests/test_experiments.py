import sqlite3

from modules.experiments import ExperimentsStore


def test_experiments_store_v5190_tables_and_writes(tmp_path):
    store = ExperimentsStore(str(tmp_path))

    store.upsert_config_snapshot(
        snapshot_hash="abc123",
        config_json={"agent_mode": "PAPER"},
        source="test",
        note="snapshot",
    )
    store.log_experiment(
        symbol="AAPL",
        regime="BULL",
        hypothesis="daily_quick_backtest_sweep",
        strategy_name="STRATEGY_MOMENTUM",
        score=12.5,
        status="paper_candidate",
        details={"ok": True},
        config_hash="abc123",
    )
    store.upsert_strategy(
        strategy_name="STRATEGY_MOMENTUM",
        strategy_version="v5.19.0",
        status="paper_candidate",
        params={"rsi": 35},
        metadata={"symbol": "AAPL"},
    )
    store.log_deployment(
        strategy_name="STRATEGY_MOMENTUM",
        strategy_version="v5.19.0",
        stage="PAPER",
        status="candidate",
        reason="daily_research_sweep",
        metrics={"total_pl": 12.5},
    )

    db = tmp_path / "experiments.db"
    assert db.exists()

    conn = sqlite3.connect(str(db))
    try:
        n_exp = conn.execute("SELECT COUNT(1) FROM experiments").fetchone()[0]
        n_reg = conn.execute("SELECT COUNT(1) FROM strategy_registry").fetchone()[0]
        n_dep = conn.execute("SELECT COUNT(1) FROM deployments").fetchone()[0]
        n_cfg = conn.execute("SELECT COUNT(1) FROM config_snapshots").fetchone()[0]
    finally:
        conn.close()

    assert n_exp >= 1
    assert n_reg >= 1
    assert n_dep >= 1
    assert n_cfg >= 1
