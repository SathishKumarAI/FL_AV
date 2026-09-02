"""The live edge fleet: machines running the global model on a camera, right now.

This is the other half of the project. `pipeline/vehicles.py` simulates a fleet to
*train* a model; this tracks real machines *running* it, so the thing federation
produces can be watched working rather than only scored.

Deliberately small, and deliberately not a database:

- **Frames live in memory and never touch disk.** A node POSTs a JPEG, the newest one
  per node is held, the previous is dropped. Nothing is written, so there is no path to
  traverse, no directory to fill, and nothing to clean up after a node goes away.
- **A node is whatever last said hello.** No registration handshake and no persistence:
  a node that stops reporting ages out of the listing, and a node that comes back is
  simply present again. Restarting the dashboard forgets the fleet, which is correct --
  the fleet is a live fact, not a record.
- **Every field a node sends is bounded before it is stored.** The registry is the trust
  boundary: the server is loopback-only today, but the whole point of this module is
  that one day it is not.

    python -m pipeline.nodes          # what is live right now
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path

from . import paths

#: A node names itself. Anything outside this is refused rather than sanitised -- a
#: silently renamed node is two nodes in the listing and one confused operator.
NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")

#: Bigger than a 640px JPEG needs, small enough that a wrong Content-Length cannot be
#: used to hold memory. One frame per node is retained, so this is the per-node ceiling.
MAX_FRAME_BYTES = 512 * 1024

#: Seconds without a heartbeat before a node is shown as gone. Nodes post at ~1 Hz, so
#: this tolerates three missed beats -- long enough to survive a GC pause, short enough
#: that a dead node does not sit on the dashboard looking healthy.
OFFLINE_AFTER = 12.0

#: How many nodes may be tracked at once. A bound, not a design limit: without it, a
#: loop that generates a fresh id per POST would grow the registry forever.
MAX_NODES = 64

_LOCK = threading.Lock()
_NODES: dict[str, dict] = {}
_FRAMES: dict[str, bytes] = {}


def _clamp(value, lo, hi, default=0.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def heartbeat(node_id: str, payload: dict, frame: bytes | None = None,
              now: float | None = None) -> dict:
    """Record one report from an edge node. Returns what was stored.

    Raises ``ValueError`` with a reason a human can act on -- the node prints it and
    keeps running, rather than dying silently on the far end of a network.
    """
    if not NODE_ID.match(node_id or ""):
        raise ValueError(f"node id {node_id!r} must match {NODE_ID.pattern}")
    if frame is not None and len(frame) > MAX_FRAME_BYTES:
        raise ValueError(f"frame is {len(frame)} bytes, the limit is {MAX_FRAME_BYTES}")

    now = time.time() if now is None else now
    # Class counts are the one open-ended field a node sends. Bound the number of keys
    # and the length of each: the detector has 13 classes, so anything past that is a
    # node sending something this dashboard was not built to show.
    counts = {}
    for name, n in list((payload.get("counts") or {}).items())[:32]:
        try:
            counts[str(name)[:32]] = int(n)
        except (TypeError, ValueError):
            continue

    record = {
        "id": node_id,
        "label": str(payload.get("label") or node_id)[:64],
        "host": str(payload.get("host") or "")[:64],
        "source": str(payload.get("source") or "")[:64],
        # Which global checkpoint this node is actually running. The point of the whole
        # panel: a node still on round 3 while the server has published round 6 is the
        # normal state during a run, and it has to be visible or the detections on
        # screen get read as the current model's.
        "model": str(payload.get("model") or "")[:64],
        "model_round": payload.get("model_round"),
        "fps": _clamp(payload.get("fps"), 0, 1000),
        "latency_ms": _clamp(payload.get("latency_ms"), 0, 60000),
        "detections": int(_clamp(payload.get("detections"), 0, 10000)),
        "counts": counts,
        "device": str(payload.get("device") or "")[:32],
        "error": str(payload.get("error") or "")[:200],
        "seen": now,
        "frames": (_NODES.get(node_id, {}).get("frames") or 0) + 1,
    }

    with _LOCK:
        if node_id not in _NODES and len(_NODES) >= MAX_NODES:
            raise ValueError(f"too many nodes ({MAX_NODES}); this one was refused")
        _NODES[node_id] = record
        if frame:
            _FRAMES[node_id] = frame
    return record


def listing(now: float | None = None) -> dict:
    """Every node the dashboard should draw, newest heartbeat first."""
    now = time.time() if now is None else now
    with _LOCK:
        rows = [dict(r) for r in _NODES.values()]
    for r in rows:
        age = now - r["seen"]
        r["age_s"] = round(age, 1)
        r["online"] = age <= OFFLINE_AFTER
        r["has_frame"] = r["id"] in _FRAMES
        r.pop("seen", None)
    rows.sort(key=lambda r: (not r["online"], r["id"]))
    online = [r for r in rows if r["online"]]
    return {
        "nodes": rows,
        "online": len(online),
        "total": len(rows),
        # Summed rather than averaged: five cameras at 12 fps is sixty frames a second
        # of real inference, which is the number that describes the fleet.
        "fleet_fps": round(sum(r["fps"] for r in online), 1),
        "detections": sum(r["detections"] for r in online),
        "offline_after": OFFLINE_AFTER,
    }


def frame(node_id: str) -> bytes | None:
    with _LOCK:
        return _FRAMES.get(node_id)


def forget(node_id: str) -> bool:
    with _LOCK:
        _FRAMES.pop(node_id, None)
        return _NODES.pop(node_id, None) is not None


def reset() -> None:
    with _LOCK:
        _NODES.clear()
        _FRAMES.clear()


# --------------------------------------------------------------------------
# What an edge node needs from the server: which model to run, and its bytes
# --------------------------------------------------------------------------
def checkpoint_dir() -> Path:
    return paths.PROJECT / "checkpoints"


def latest_model() -> dict:
    """The newest global checkpoint, described well enough for a node to cache it.

    ``global_last.pt`` is ignored on purpose -- it duplicates the highest round under a
    name that carries no round number, and a node caching by name would never notice it
    change. The same reasoning as ``holdout.checkpoints``.
    """
    rounds = sorted(checkpoint_dir().glob("global_round_*.pt"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not rounds:
        return {"available": False, "reason": "no global checkpoint yet; run the federation"}
    newest = rounds[-1]
    data = newest.read_bytes()
    return {
        "available": True,
        "name": newest.name,
        "round": int(newest.stem.rsplit("_", 1)[-1]),
        "bytes": len(data),
        # Nodes cache on this, not on the round number: a re-run rewrites round 1 with
        # different weights under the same name, and a node keyed on the name alone
        # would keep serving the previous run's model without ever being wrong-looking.
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def model_bytes() -> tuple[bytes, str] | None:
    info = latest_model()
    if not info.get("available"):
        return None
    return (checkpoint_dir() / info["name"]).read_bytes(), info["name"]


def main(argv=None) -> int:
    info = latest_model()
    print(f"checkpoints: {checkpoint_dir()}")
    print(f"latest: {info}")
    live = listing()
    print(f"\n{live['online']} of {live['total']} node(s) online, "
          f"{live['fleet_fps']} fps across the fleet")
    for n in live["nodes"]:
        mark = "live" if n["online"] else f"gone {n['age_s']}s"
        print(f"  {n['id']:<16} {mark:<12} {n['fps']:>5.1f} fps  "
              f"{n['latency_ms']:>6.1f} ms  round {n['model_round']}")
    if not live["nodes"]:
        print("  (none — this registry is in-memory, so only nodes reporting to a "
              "RUNNING dashboard appear here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
