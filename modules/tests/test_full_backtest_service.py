from modules.research.full_backtest_service import run_full_backtest_service


class _DF:
    def __init__(self, empty=False):
        self.empty = empty


class _Cfg:
    def sections(self):
        return ["STRATEGY_A", "STRATEGY_B"]


class _DB:
    def __init__(self):
        self.saved = []

    def rebuild_backtest_table(self, _strategies):
        return None

    def get_all_symbols(self):
        return ["AAPL", "MSFT"]

    def get_history(self, _sym, _limit):
        return _DF(empty=False)

    def save_backtest_result(self, row):
        self.saved.append(dict(row))


def test_full_backtest_service_runs_and_persists():
    db = _DB()

    def _sim(_df, strat, sym):
        return (10.0 if strat == "A" else 5.0, 3)

    out = run_full_backtest_service(_Cfg(), db, simulate_strategy=_sim)

    assert out.get("ok") is True
    assert out.get("count") == 2
    assert len(db.saved) == 2
    assert all("best_strategy" in r for r in db.saved)
    assert all("timeframe" in r and "results_json" in r for r in db.saved)
    assert all("best_strategy_fingerprint" in r for r in db.saved)


class _DBSaveFail(_DB):
    def save_backtest_result(self, _row):
        raise RuntimeError("boom")


def test_full_backtest_service_returns_rows_and_save_failure_counts():
    db = _DBSaveFail()

    def _sim(_df, strat, _sym):
        return (1.0 if strat == "A" else -0.5, 1)

    out = run_full_backtest_service(_Cfg(), db, simulate_strategy=_sim)

    assert out.get("ok") is True
    assert out.get("count") == 2
    assert out.get("saved_count") == 0
    assert out.get("save_failures") == 2
    rows = out.get("rows")
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all("symbol" in r and "best_strategy" in r for r in rows)
    assert all("trade_count" in r and "results_json" in r for r in rows)


class _DBNoArgRebuild(_DB):
    def rebuild_backtest_table(self):
        return None


def test_full_backtest_service_supports_noarg_rebuild_signature():
    db = _DBNoArgRebuild()

    def _sim(_df, strat, _sym):
        return (2.0 if strat == "A" else 1.0, 2)

    out = run_full_backtest_service(_Cfg(), db, simulate_strategy=_sim, rebuild_table=True)

    assert out.get("ok") is True
    assert out.get("count") == 2
    assert out.get("saved_count") == 2


class _CfgDup:
    def sections(self):
        return ["STRATEGY_A", "STRATEGY_B"]

    def has_section(self, sec):
        return sec in ("STRATEGY_A", "STRATEGY_B")

    def __getitem__(self, sec):
        return {"enabled": "True", "priority": "1"}


def test_full_backtest_service_reports_strategy_fingerprint_collisions():
    db = _DB()

    out = run_full_backtest_service(_CfgDup(), db, simulate_strategy=lambda _d, s, _y: (1.0 if s == "A" else 0.9, 1))

    assert out.get("ok") is True
    assert isinstance(out.get("strategy_fingerprints"), dict)
    groups = out.get("duplicate_fingerprint_groups") or []
    assert groups and sorted(groups[0]) == ["A", "B"]
