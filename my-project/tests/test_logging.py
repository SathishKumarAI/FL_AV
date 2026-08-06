"""Tests for rotating, bounded logging setup."""
import logging
import os
from logging.handlers import RotatingFileHandler

from utils.logging_setup import configure_logging


def test_file_handler_is_rotating(tmp_path):
    log_file = tmp_path / "sub" / "test.log"
    lg = configure_logging("fl_av_test_rotating", str(log_file))
    handlers = [h for h in lg.handlers if isinstance(h, RotatingFileHandler)]
    assert handlers, "expected a RotatingFileHandler"
    assert handlers[0].maxBytes > 0
    assert handlers[0].backupCount >= 1
    # The handler created its parent directory.
    assert log_file.parent.is_dir()
    lg.info("hello")
    # One file per process: RotatingFileHandler is not multi-process safe, so the
    # pid is folded into the name and callers glob for it.
    written = list(log_file.parent.glob("test.*.log"))
    assert written == [log_file.with_name(f"test.{os.getpid()}.log")]


def test_no_duplicate_handlers_on_repeat_calls():
    a = configure_logging("fl_av_test_dup", None)
    n = len(a.handlers)
    b = configure_logging("fl_av_test_dup", None)
    assert a is b
    assert len(b.handlers) == n  # not stacked on repeat configuration


def test_does_not_propagate_to_root():
    lg = configure_logging("fl_av_test_propagate", None)
    assert lg.propagate is False


# ------------------------------------------------------- construction is inert
def test_building_a_metrics_logger_does_not_touch_the_file():
    """A strategy probe wiped rounds 1-5 of a live six-round run in its final
    minute, because constructing one truncated logs/metrics.csv. Only a real row
    may start the file."""
    import csv as _csv
    import tempfile
    from pathlib import Path as _Path

    from utils.metrics_logger import MetricsLogger

    with tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "logs" / "metrics.csv"
        path.parent.mkdir(parents=True)
        path.write_text("round,stage\n5,evaluate\n")

        MetricsLogger(str(path))
        MetricsLogger(str(path))
        assert path.read_text() == "round,stage\n5,evaluate\n", (
            "constructing a logger must not truncate a file another run is writing")

        logger = MetricsLogger(str(path))
        logger.log_round(1, "evaluate", {"mAP50": 0.5}, num_clients=2, loss=0.1)
        rows = list(_csv.DictReader(path.read_text().splitlines()))
        assert len(rows) == 1 and rows[0]["round"] == "1", "the first row starts a fresh file"
