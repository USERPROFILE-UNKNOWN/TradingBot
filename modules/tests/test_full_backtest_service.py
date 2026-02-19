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
