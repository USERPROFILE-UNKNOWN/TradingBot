"""Repository layer for persistence (v5.16.x).

These repositories encapsulate the existing SQL behavior so DataManager can
gradually shrink into a façade (v5.16.2).
"""

from .trades_repo import TradesRepo
from .history_repo import HistoryRepo
from .decisions_repo import DecisionsRepo
from .backtest_repo import BacktestRepo
from .candidates_repo import CandidatesRepo
from .agent_repo import AgentRepo

__all__ = [
    "TradesRepo",
    "HistoryRepo",
    "DecisionsRepo",
    "BacktestRepo",
    "CandidatesRepo",
    "AgentRepo",
]
