"""Where the seconds of a round actually went.

Phase 0 of the plan exists because "27 % mean GPU utilisation" is compatible with two
different worlds that have opposite fixes: clients serialised on one card, or a
dataloader starving the GPU inside ``train()``. The mean cannot tell them apart. The
timestamps can, and they are already in the logs.

Nothing here adds logging to my-project. Every boundary below is a pair of lines this
project has written since before the profiler existed, which is why the 3 296 s
reference run of 2026-08-06 can be profiled after the fact rather than re-run.

Two numbers decide phase 1, and they are measured separately rather than one inferred
from the other:

* ``train_share``  -- wall clock spent inside train(), over wall clock. High means the
  overhead is *inside* training, so the levers are cache and dataloader workers.
* ``max_concurrent`` -- the most client episodes overlapping at any instant. 1 proves
  serialisation, whatever train_share says, and makes num-gpus the whole result.

    python -m pipeline.profile
    python -m pipeline.profile --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from . import logparse, paths

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - \w+ - \[(\w+)\] (.*)$")

# marker substring -> (phase, "open" | "close"). One line can drive two phases: the
# line that ends training is the same line that starts serialising the result, so
# every entry is tested rather than stopping at the first hit.
CLIENT_MARKERS = [
    ("Creating FlowerClient instance", "construct", "open"),
    ("YOLO model loaded successfully", "construct", "close"),
    ("Received weights with checksum", "weights_in", "open"),
    ("Received evaluation weights with checksum", "weights_in", "open"),
    ("Successfully applied received weights", "weights_in", "close"),
    ("Setting evaluation weights", "weights_in", "close"),
    ("Starting local training with batch_id", "train", "open"),
    ("Training done", "train", "close"),
    ("Training done", "weights_out", "open"),
    ("Sending back weights with checksum", "weights_out", "close"),
    ("Starting evaluation with batch_id", "evaluate", "open"),
    ("Evaluation done", "evaluate", "close"),
]

SERVER_MARKERS = [
    ("Aggregating", "aggregate", "open"),
    ("Aggregated parameters with checksum", "aggregate", "close"),
    ("Aggregated parameters with checksum", "checkpoint", "open"),
    ("Saved global checkpoint", "checkpoint", "close"),
]

# An episode is one client's turn on the card: constructed, given weights, trained or
# evaluated, handed back. Overlapping episodes are what concurrency means here.
EPISODE_OPEN = "Creating FlowerClient instance"
EPISODE_CLOSE = ("Sending back weights with checksum", "Evaluation done")

PHASES = ["construct", "weights_in", "train", "weights_out", "evaluate",
          "aggregate", "checkpoint"]


def _parse_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def intervals(text: str, markers: list[tuple[str, str, str]]) -> dict[str, list[tuple[float, float]]]:
    """Pair open/close markers into [start, end) intervals per phase.

    A phase reopened before it closed keeps the *later* open: a client that crashed
    mid-round leaves a dangling marker, and carrying the stale one forward would
    attribute the crash gap to whatever phase happened to be open.
    """
    out: dict[str, list[tuple[float, float]]] = {p: [] for p in PHASES}
    open_at: dict[str, float] = {}
    for line in text.splitlines():
        m = TS.match(line)
        if not m:
            continue
        t, body = _parse_ts(m.group(1)), m.group(3)
        for needle, phase, side in markers:
            if needle not in body:
                continue
            if side == "open":
                open_at[phase] = t
            elif (start := open_at.pop(phase, None)) is not None:
                out[phase].append((start, t))
    return out


def episodes(text: str) -> list[tuple[float, float]]:
    """One client's turn on the card, start to hand-back."""
    out, start = [], None
    for line in text.splitlines():
        m = TS.match(line)
        if not m or m.group(2) != "Client":
            continue
        t, body = _parse_ts(m.group(1)), m.group(3)
        if EPISODE_OPEN in body:
            start = t
        elif start is not None and any(c in body for c in EPISODE_CLOSE):
            out.append((start, t))
            start = None
    return out


def union_seconds(spans: list[tuple[float, float]]) -> float:
    """Wall clock covered by at least one span. Not the sum: spans can overlap."""
    total, end = 0.0, None
    for s, e in sorted(spans):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def max_overlap(spans: list[tuple[float, float]]) -> int:
    """Most spans open at once. 1 means serialised."""
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    best = cur = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def stamps_of(text: str) -> list[float]:
    return [_parse_ts(m.group(1)) for m in map(TS.match, text.splitlines()) if m]


def profile(server_log: Path | None = None, log_dir: Path | None = None) -> dict:
    """Per-phase seconds for one run, from the logs it already wrote."""
    server = server_log or logparse.latest_run_log(log_dir)
    if server is None:
        return {"error": "no server log that aggregated a round -- has a federation run?"}

    server_text = server.read_text(errors="replace")
    server_stamps = stamps_of(server_text)
    if not server_stamps:
        return {"error": f"no timestamped lines in {server.name}"}
    t0, t1 = min(server_stamps), max(server_stamps)

    # Client logs of *this* run only, decided by what the logs say rather than by when
    # the filesystem last touched them. Logs are named per process and accumulate, and
    # the mtime window this started with swept in the previous arm of a before/after
    # comparison: two runs four minutes apart, 48 episodes, a wall clock spanning both,
    # and a "2x faster" reading that was two runs added together.
    clients = []
    for f in logparse.iter_logs("client*.log", log_dir):
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")
        s = stamps_of(text)
        # A client of this run started while the server was still logging, so its first
        # line falls inside the server's own span. Both ends are tight: a minute of
        # slack at either end is enough to swallow the next arm of a before/after
        # comparison, which is the measurement this exists to make.
        if s and t0 - 5 <= min(s) <= t1:
            clients.append((f, text))

    spans: dict[str, list[tuple[float, float]]] = {p: [] for p in PHASES}
    eps: list[tuple[float, float]] = []
    stamps: list[float] = list(server_stamps)

    for f, text in clients:
        for phase, got in intervals(text, CLIENT_MARKERS).items():
            spans[phase] += got
        eps += episodes(text)
        stamps += stamps_of(text)

    for phase, got in intervals(server_text, SERVER_MARKERS).items():
        spans[phase] += got

    wall = max(stamps) - min(stamps)
    busy = union_seconds(eps)
    breakdown = {p: round(union_seconds(spans[p]), 1) for p in PHASES}
    train_share = breakdown["train"] / wall if wall else 0.0

    return {
        "server_log": str(server),
        "client_logs": [str(f) for f, _ in clients],
        "wall_s": round(wall, 1),
        "phases": breakdown,
        # Union, so this is honest whether or not episodes overlap. Whatever is left is
        # Ray scheduling, process teardown and genuine idle -- named as unaccounted
        # rather than modelled, because the timestamps do not say which.
        "unaccounted_s": round(wall - busy, 1),
        "episodes": len(eps),
        "max_concurrent": max_overlap(eps),
        "train_share": round(train_share, 3),
    }


def verdict(p: dict) -> list[str]:
    """Which of the two worlds the run was in. Both checks, independently."""
    out = []
    if p["max_concurrent"] <= 1 and p["episodes"] > 1:
        out.append(f"[SERIALISED] {p['episodes']} client episodes, never more than "
                   f"{p['max_concurrent']} at a time -- one client owns the card. "
                   f"num-gpus < 1.0 is the lever, and it is a no-op mathematically.")
    else:
        out.append(f"[CONCURRENT] up to {p['max_concurrent']} client episodes overlap; "
                   f"packing more clients will not be the whole win.")
    if p["train_share"] >= 0.85:
        out.append(f"[IN-TRAINING] {p['train_share']:.0%} of the wall clock is inside "
                   f"train(). The overhead is the data path, not orchestration: "
                   f"cache and dataloader workers are the levers.")
    else:
        out.append(f"[AROUND-TRAINING] only {p['train_share']:.0%} of the wall clock is "
                   f"inside train(); {100 * (1 - p['train_share']):.0f}% is setup, "
                   f"aggregation and idle. Fix that before touching the data path.")
    return out


def render(p: dict) -> str:
    if "error" in p:
        return f"profile: {p['error']}"
    wall = p["wall_s"]
    lines = [f"{Path(p['server_log']).name} -- {len(p['client_logs'])} client log(s), "
             f"{p['episodes']} episodes, wall {wall:.0f}s",
             "",
             f"{'phase':<14}{'seconds':>10}{'share':>9}"]
    for phase, secs in p["phases"].items():
        lines.append(f"{phase:<14}{secs:>10.1f}{secs / wall if wall else 0:>9.1%}")
    lines.append(f"{'unaccounted':<14}{p['unaccounted_s']:>10.1f}"
                 f"{p['unaccounted_s'] / wall if wall else 0:>9.1%}")
    return "\n".join(lines + [""] + verdict(p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server-log", type=Path, help="profile this run instead of the last one")
    ap.add_argument("--json", action="store_true", help="also write .state/profile-<stamp>.json")
    a = ap.parse_args(argv)

    p = profile(a.server_log)
    print(render(p))
    if "error" in p:
        return 1
    if a.json:
        paths.STATE.mkdir(parents=True, exist_ok=True)
        stamp = Path(p["server_log"]).stem.replace("server.", "")
        out = paths.STATE / f"profile-{stamp}.json"
        out.write_text(json.dumps(p, indent=2))
        print(f"\nwrote {out}")
    return 0


def demo() -> None:
    """Self-check: a hand-computable two-episode run."""
    text = "\n".join([
        "2026-08-06 00:00:00,000 - INFO - [Client] Creating FlowerClient instance from client_fn.",
        "2026-08-06 00:00:01,000 - INFO - [Client] YOLO model loaded successfully.",
        "2026-08-06 00:00:01,000 - INFO - [Client] Received weights with checksum: 1.0",
        "2026-08-06 00:00:02,000 - INFO - [Client] Successfully applied received weights to model",
        "2026-08-06 00:00:02,000 - INFO - [Client] Starting local training with batch_id=1, local_epochs=4",
        "2026-08-06 00:00:12,000 - INFO - [Client] 1 Training done. metrics={}",
        "2026-08-06 00:00:13,000 - INFO - [Client] Sending back weights with checksum: 2.0",
        "2026-08-06 00:00:20,000 - INFO - [Client] Creating FlowerClient instance from client_fn.",
        "2026-08-06 00:00:20,000 - INFO - [Client] YOLO model loaded successfully.",
        "2026-08-06 00:00:20,000 - INFO - [Client] Starting evaluation with batch_id=1",
        "2026-08-06 00:00:25,000 - INFO - [Client] Evaluation done. Loss=0.1, metrics={}",
    ])
    got = intervals(text, CLIENT_MARKERS)
    assert union_seconds(got["train"]) == 10.0, got["train"]
    assert union_seconds(got["construct"]) == 1.0, got["construct"]
    assert union_seconds(got["weights_out"]) == 1.0, got["weights_out"]
    assert union_seconds(got["evaluate"]) == 5.0, got["evaluate"]

    eps = episodes(text)
    assert len(eps) == 2 and max_overlap(eps) == 1, eps
    # Two episodes, 13 s and 5 s, with a 7 s gap: 25 s wall, 18 s busy.
    assert union_seconds(eps) == 18.0, eps

    # Overlap is counted, not assumed away.
    assert max_overlap([(0.0, 10.0), (5.0, 15.0), (20.0, 30.0)]) == 2
    assert union_seconds([(0.0, 10.0), (5.0, 15.0)]) == 15.0
    print("profile self-check OK")


if __name__ == "__main__":
    sys.exit(main())
