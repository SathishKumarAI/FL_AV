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


def test_a_relative_log_path_lands_in_the_project_not_the_working_directory(tmp_path,
                                                                            monkeypatch):
    """The bug: importing a module scattered log files wherever you were standing.

    `pytest my-project/tests` from the repo root imports `my_project.server_app` at
    collection, which configured `logs/server.log` relative to the CWD and created an
    empty `logs/server.<pid>.log` in the repo root. That file looked newer than the
    real federation's log, and `pipeline.verify` then reported "need >=2 rounds to
    tell, saw 0" straight after a six-round run had succeeded.

    So: chdir somewhere else entirely and assert nothing appears there.
    """
    from pathlib import Path

    import utils.logging_setup as ls

    monkeypatch.delenv("FL_AV_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    lg = configure_logging("fl_av_test_anchor", "logs/anchor_probe.log")
    handler = next(h for h in lg.handlers if isinstance(h, RotatingFileHandler))
    written = Path(handler.baseFilename)
    try:
        lg.info("hello")
        assert written.parent == ls.project_root() / "logs", written
        assert not (tmp_path / "logs").exists(), "a relative path still followed the CWD"
    finally:
        handler.close()
        lg.handlers.clear()
        written.unlink(missing_ok=True)


def test_the_run_root_wins_over_this_files_location(tmp_path, monkeypatch):
    """`flwr run` executes a *copy*, so `__file__` is not the checkout.

        Successfully installed my-project to ~/.flwr/apps/flower.my-project.1.0.0.480fd449

    Anchoring on `__file__` alone put the federation's logs inside that copy, where
    the CI smoke's `glob("logs/server.*.log")` could not see them, and the run was
    then judged to have aggregated nothing: `expected >=2 aggregate checksums, got []`.
    `FL_AV_DATA_ROOT` names the checkout that owns the run, and it wins.
    """
    import utils.logging_setup as ls

    monkeypatch.setenv("FL_AV_DATA_ROOT", str(tmp_path / "checkout"))
    assert ls.project_root() == tmp_path / "checkout"


def test_a_flwr_app_copy_is_never_the_log_root(tmp_path, monkeypatch):
    """With no `FL_AV_DATA_ROOT`, a copy under `.flwr` falls back to the CWD.

    Writing into flwr's app cache is the worst option available: the run looks fine
    and the evidence lands where nobody will look for it.
    """
    from pathlib import Path

    import utils.logging_setup as ls

    monkeypatch.delenv("FL_AV_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        ls, "__file__",
        str(tmp_path / ".flwr" / "apps" / "flower.my-project.1.0.0.abc" / "utils" / "logging_setup.py"))
    monkeypatch.chdir(tmp_path)
    assert ls.project_root() == Path.cwd()


def test_an_absolute_log_path_is_left_alone(tmp_path):
    """The anchor must not capture a caller who said exactly where they wanted it."""
    from pathlib import Path

    target = tmp_path / "elsewhere" / "explicit.log"
    lg = configure_logging("fl_av_test_absolute", str(target))
    handler = next(h for h in lg.handlers if isinstance(h, RotatingFileHandler))
    assert Path(handler.baseFilename).parent == target.parent


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
