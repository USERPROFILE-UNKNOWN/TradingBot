from modules.persistence.db_runtime import DbRuntime
from modules.persistence.repositories.backtest_repo import BacktestRepo


def test_save_backtest_result_adds_dynamic_strategy_columns(tmp_path):
    runtime = DbRuntime(str(tmp_path))
    runtime.init_split_connections()

    repo = BacktestRepo(runtime, split_mode=True)
    repo.ensure_backtest_table()

    row = {
        "symbol": "AAPL",
        "best_strategy": "THE_GENERAL",
        "best_profit": 123.45,
        "timestamp": "2026-02-18 23:00:00",
        "PL_THE_GENERAL": 123.45,
        "Trades_THE_GENERAL": 12,
    }
    repo.save_backtest_result(row)

    df = repo.get_backtest_data("AAPL")
    assert not df.empty
    assert "PL_THE_GENERAL" in df.columns
    assert "Trades_THE_GENERAL" in df.columns
    assert float(df.iloc[0]["PL_THE_GENERAL"]) == 123.45
    assert int(df.iloc[0]["Trades_THE_GENERAL"]) == 12


def test_save_backtest_result_maps_stable_legacy_columns(tmp_path):
    runtime = DbRuntime(str(tmp_path))
    runtime.init_split_connections()

    repo = BacktestRepo(runtime, split_mode=True)
    repo.ensure_backtest_table()

    row = {
        "symbol": "MSFT",
        "best_strategy": "BREAKOUT",
        "best_profit": 88.0,
        "timestamp": "2026-02-19 01:23:45",
        "PL_BREAKOUT": 88.0,
        "Trades_BREAKOUT": 4,
    }
    repo.save_backtest_result(row)

    df = repo.get_backtest_data("MSFT")
    assert not df.empty
    first = df.iloc[0]
    assert first["strategy"] == "BREAKOUT"
    assert float(first["total_profit"]) == 88.0
    assert first["tested_at"] == "2026-02-19 01:23:45"
    assert first["timeframe"] == "1Min"
    assert "best_strategy_fingerprint" in df.columns
    assert "strategy_fingerprints_json" in df.columns
    assert isinstance(first["results_json"], str) and "PL_BREAKOUT" in first["results_json"]
