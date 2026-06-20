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
    assert log_file.exists()


def test_no_duplicate_handlers_on_repeat_calls():
    a = configure_logging("fl_av_test_dup", None)
    n = len(a.handlers)
    b = configure_logging("fl_av_test_dup", None)
    assert a is b
    assert len(b.handlers) == n  # not stacked on repeat configuration


def test_does_not_propagate_to_root():
    lg = configure_logging("fl_av_test_propagate", None)
    assert lg.propagate is False
