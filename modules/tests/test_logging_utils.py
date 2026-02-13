import logging

from modules.logging_utils import _ContextDefaultsFilter, get_component_logger


def test_context_defaults_filter_populates_missing_fields():
    filt = _ContextDefaultsFilter()
    rec = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    assert filt.filter(rec) is True
    assert rec.component == "app"
    assert rec.symbol == "-"
    assert rec.order_id == "-"
    assert rec.strategy == "-"


def test_component_logger_injects_component_field(caplog):
    log = get_component_logger("tb.test", "startup")

    with caplog.at_level(logging.INFO):
        log.info("boot")

    assert any(getattr(r, "component", "") == "startup" for r in caplog.records)
