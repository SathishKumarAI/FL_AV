"""Turn my-project's existing log lines into structured events.

Deliberately parsing markers that are *already* emitted rather than adding logging to
my-project. The markers below are load-bearing: the round-over-round aggregate
checksum is the only thing that distinguishes a federation that learns from one that
returns the weights it was handed (the B4 bug), and the per-vehicle sent/received
pair is what makes that legible in the fleet view.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

NUM = r"(-?[\d.]+(?:e[+-]?\d+)?)"

RE_AGG = re.compile(rf"Aggregated parameters with checksum: {NUM}")
RE_SENT_GLOBAL = re.compile(rf"Sending parameters with checksum: {NUM}")
RE_RECV = re.compile(rf"Received weights with checksum: {NUM}")
RE_SEND_BACK = re.compile(rf"Sending back weights with checksum: {NUM}")
RE_TRAINING = re.compile(r"Starting local training with batch_id=(\d+), local_epochs=(\d+)")
RE_ASSIGN = re.compile(r"Assigning batch_id=(\d+) to client (\S+) in round=(\d+)")
RE_DEVICE = re.compile(r"Initializing FlowerClient with model=\S+ on (\S+)")
RE_NOOP = re.compile(r"fewer than the (\d+) needed for one optimizer step")
RE_ROUND_DONE = re.compile(r"Run finished (\d+) round\(s\) in ([\d.]+)s")


@dataclass
class Event:
    kind: str
    value: float | str | None = None
    extra: dict | None = None


def parse_line(line: str) -> Event | None:
    """One log line -> one event, or None if the line says nothing structured."""
    for regex, kind in (
        (RE_AGG, "aggregate_checksum"),
        (RE_SENT_GLOBAL, "global_sent_checksum"),
        (RE_RECV, "client_received_checksum"),
        (RE_SEND_BACK, "client_sent_checksum"),
    ):
        m = regex.search(line)
        if m:
            return Event(kind, float(m.group(1)))

    if m := RE_TRAINING.search(line):
        return Event("training_start", int(m.group(1)), {"local_epochs": int(m.group(2))})
    if m := RE_ASSIGN.search(line):
        return Event("assignment", int(m.group(1)), {"client": m.group(2), "round": int(m.group(3))})
    if m := RE_DEVICE.search(line):
        return Event("device", m.group(1))
    if m := RE_NOOP.search(line):
        # The round will finish and change nothing. Surfacing it is the whole point:
        # identical checksums otherwise look exactly like the B4 bug.
        return Event("no_optimizer_step", int(m.group(1)))
    if m := RE_ROUND_DONE.search(line):
        return Event("run_finished", float(m.group(2)), {"rounds": int(m.group(1))})
    return None


def parse_text(text: str) -> list[Event]:
    return [e for e in (parse_line(l) for l in text.splitlines()) if e]


def read_metrics_csv(path: Path) -> list[dict]:
    """logs/metrics.csv -> rows with numeric fields coerced."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out = {}
            for k, v in row.items():
                if v in (None, ""):
                    out[k] = None
                else:
                    try:
                        out[k] = float(v) if k not in ("stage",) else v
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def iter_logs(pattern: str, log_dir: Path | None = None):
    """Matching log files across every directory a run can write to."""
    from . import paths
    dirs = [log_dir] if log_dir is not None else paths.log_dirs()
    for d in dirs:
        if d and d.is_dir():
            yield from sorted(d.glob(pattern))


def latest_log(pattern: str, log_dir: Path | None = None) -> Path | None:
    """The most recently written matching log, or None.

    Logs are named per process, so they accumulate across runs. Reading all of them
    concatenates several runs into one sequence -- eleven checksums for a three-round
    run -- and the checksum criterion then judges a mixture rather than this run.
    """
    files = [f for f in iter_logs(pattern, log_dir) if f.is_file()]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def aggregate_checksums(log_dir: Path | None = None, all_runs: bool = False) -> list[float]:
    """This run's per-round global checksums, in order.

    Equal consecutive values mean nothing is being learned, whatever the metrics say
    -- which is why it matters that these come from one run. Pass ``all_runs=True``
    for the historical sequence.
    """
    files = list(iter_logs("server*.log", log_dir)) if all_runs else \
        [f for f in [latest_log("server*.log", log_dir)] if f]
    out: list[float] = []
    for f in files:
        out += [e.value for e in parse_text(f.read_text(errors="replace"))
                if e.kind == "aggregate_checksum"]
    return out


def federation_learned(log_dir: Path | None = None) -> tuple[bool, str]:
    """The B4 guard, as a reusable check."""
    cs = aggregate_checksums(log_dir)
    if len(cs) < 2:
        return False, f"need >=2 rounds to tell, saw {len(cs)}"
    if len(set(cs)) != len(cs):
        return False, f"global weights did not change between rounds: {cs}"
    return True, f"weights moved every round: {cs}"


def demo() -> None:
    """Self-check against real captured lines from this project's logs."""
    sample = """
2026-08-05 - INFO - [Server] Aggregated parameters with checksum: -1032.5395936965942
2026-08-05 - INFO - [Client] Received weights with checksum: 698.6960870027542
2026-08-05 - INFO - [Client] Sending back weights with checksum: 679.3249893784523
2026-08-05 - INFO - [Client] Starting local training with batch_id=9, local_epochs=1
2026-08-05 - INFO - [Server] Assigning batch_id=4 to client 4520368180988013936 in round=2
2026-08-05 - INFO - [Client] Initializing FlowerClient with model=models/yolov8s.pt on cuda:0
2026-08-05 - WARNING - [Client] batch=16 over 10 images x 1 epoch(s) gives 1 batch(es), fewer than the 4 needed for one optimizer step.
INFO :      Run finished 2 round(s) in 919.87s
"""
    events = parse_text(sample)
    kinds = [e.kind for e in events]
    assert kinds == ["aggregate_checksum", "client_received_checksum", "client_sent_checksum",
                     "training_start", "assignment", "device", "no_optimizer_step",
                     "run_finished"], kinds
    assert events[0].value == -1032.5395936965942       # negative + exponent-free
    assert events[3].value == 9 and events[3].extra["local_epochs"] == 1
    assert events[4].extra["round"] == 2
    assert events[5].value == "cuda:0"
    assert events[7].extra["rounds"] == 2 and events[7].value == 919.87
    print("logparse self-check OK:", kinds)


if __name__ == "__main__":
    demo()
