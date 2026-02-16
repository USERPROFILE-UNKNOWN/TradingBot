import sys
import types

sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from modules.database import DataManager


class _CaptureLogger:
    def __init__(self):
        self.calls = []

    def info(self, msg, extra=None):
        self.calls.append((msg, extra))


def test_datamanager_log_uses_structured_callback_when_available():
    dm = DataManager.__new__(DataManager)
    calls = []

    def _cb(msg, **context):
        calls.append((msg, context))

    dm._log_cb = _cb
    dm._logger = _CaptureLogger()

    dm._log("hello", symbol="AAPL", category="DB")

    assert len(calls) == 1
    assert calls[0][0] == "hello"
    assert calls[0][1]["component"] == "db"
    assert calls[0][1]["symbol"] == "AAPL"


def test_datamanager_log_falls_back_to_legacy_single_arg_callback():
    dm = DataManager.__new__(DataManager)
    calls = []

    def _legacy(msg):
        calls.append(msg)

    dm._log_cb = _legacy
    dm._logger = _CaptureLogger()

    dm._log("legacy", order_id="oid-1")

    assert calls == ["legacy"]


def test_datamanager_log_uses_module_logger_when_no_callback():
    dm = DataManager.__new__(DataManager)
    dm._log_cb = None
    logger = _CaptureLogger()
    dm._logger = logger

    dm._log("db warning", strategy="MOMO")

    assert logger.calls == [("db warning", {"component": "db", "mode": "OFF", "strategy": "MOMO"})]
