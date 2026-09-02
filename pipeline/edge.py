"""An edge node: one machine, one camera, the current global model, live.

Run this on each test machine. It pulls the newest global checkpoint from the
dashboard, runs it on a camera, and reports what it sees. Nothing here trains -- a
camera stream has no labels, and training on the model's own guesses is a confirmation
loop, not federated learning. Training stays on the labelled BDD100K shards.

    # on the machine running the dashboard
    python -m pipeline.server

    # on each test machine (or five terminals on one)
    python -m pipeline.edge --id cam-1 --server http://192.168.1.10:8800
    python -m pipeline.edge --id cam-2 --server http://192.168.1.10:8800 --camera 1

    # no camera attached? still shows the whole path end to end:
    python -m pipeline.edge --id cam-3 --source synthetic

The point of the panel it feeds is the round number. A node keeps running the model it
downloaded until a newer one is published, so during a federation the fleet is visibly
split across rounds -- and detections on screen belong to the round the node reports,
not to whatever the server has finished aggregating.

Depends only on what the project already installs: ultralytics, opencv (a hard
dependency of ultralytics), and the standard library.
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".cache" / "federated-yolov8" / "edge"


# --------------------------------------------------------------------------
# Talking to the dashboard
# --------------------------------------------------------------------------
def _get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310 (operator-supplied)
        return r.read()


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310
        return json.loads(r.read() or b"{}")


def fetch_model(server: str, have: str | None) -> tuple[Path, dict] | None:
    """Download the newest global checkpoint unless we already hold that exact one.

    Keyed on the content hash, not the round number: re-running a federation rewrites
    global_round_1.pt with different weights under the same name, and a node caching by
    name would serve the previous run's model forever without ever looking stale.
    """
    info = json.loads(_get(f"{server}/api/model"))
    if not info.get("available"):
        return None
    if info["sha256"] == have:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{info['sha256']}-{info['name']}"
    if not path.exists():
        path.write_bytes(_get(f"{server}/api/model-file", timeout=120))
    return path, info


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def open_camera(index: int):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"no camera at index {index}. Try --camera 1, or --source synthetic to "
            f"run the whole path without one.")
    return cap


def synthetic_frame(i: int):
    """A moving shape, so a node with no camera still exercises every other step.

    It is not a road, and the detector will mostly find nothing in it. That is the
    honest behaviour: this proves transport, model loading and the dashboard, and it
    does not pretend to prove detection quality.
    """
    import numpy as np
    h, w = 480, 640
    frame = np.full((h, w, 3), 32, dtype=np.uint8)
    x = int((i * 7) % (w - 120))
    frame[180:300, x:x + 120] = (60, 140, 220)
    return frame


# --------------------------------------------------------------------------
def run(args) -> int:
    from ultralytics import YOLO
    import cv2

    server = args.server.rstrip("/")
    label = args.label or args.id
    host = socket.gethostname()
    print(f"[edge] {args.id} on {host} -> {server}")

    yolo, model_info, have = None, {}, None
    cap = None if args.source == "synthetic" else open_camera(args.camera)
    frames = 0
    last_frame_post = 0.0
    fps, latency_ms = 0.0, 0.0

    try:
        while True:
            started = time.perf_counter()

            # A new global model may be published at any time; check on a slow cadence
            # so a five-node fleet does not hammer the dashboard with hash checks.
            if frames % args.check_every == 0:
                try:
                    got = fetch_model(server, have)
                    if got:
                        path, model_info = got
                        yolo = YOLO(str(path))
                        have = model_info["sha256"]
                        print(f"[edge] loaded {model_info['name']} "
                              f"(round {model_info['round']}, {model_info['sha256']})")
                except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                    print(f"[edge] model check failed: {e}")

            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("[edge] camera returned no frame; stopping")
                    break
            else:
                frame = synthetic_frame(frames)

            counts, detections = {}, 0
            if yolo is not None:
                t0 = time.perf_counter()
                result = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                      device=args.device, verbose=False)[0]
                latency_ms = (time.perf_counter() - t0) * 1000
                frame = result.plot()
                names = result.names or {}
                for c in result.boxes.cls.tolist() if result.boxes is not None else []:
                    name = names.get(int(c), str(int(c)))
                    counts[name] = counts.get(name, 0) + 1
                detections = sum(counts.values())

            payload = {
                "label": label, "host": host, "source": args.source,
                "model": model_info.get("name", ""),
                "model_round": model_info.get("round"),
                "fps": round(fps, 2), "latency_ms": round(latency_ms, 2),
                "detections": detections, "counts": counts,
                "device": str(args.device),
                "error": "" if yolo is not None else "waiting for a global checkpoint",
            }

            # The picture is the expensive part of the payload, so it goes at its own
            # slower cadence. Telemetry stays at full rate: a node that has stopped
            # detecting must show that immediately, not at the next frame upload.
            now = time.time()
            if now - last_frame_post >= args.frame_every:
                small = cv2.resize(frame, (args.frame_width,
                                           int(frame.shape[0] * args.frame_width / frame.shape[1])))
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    payload["frame"] = base64.b64encode(buf.tobytes()).decode()
                    last_frame_post = now

            try:
                _post_json(f"{server}/api/node/{args.id}", payload)
            except (urllib.error.URLError, OSError) as e:
                print(f"[edge] heartbeat failed: {e}")

            frames += 1
            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            if args.max_fps:
                slack = (1.0 / args.max_fps) - elapsed
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[edge] stopped")
    finally:
        if cap is not None:
            cap.release()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="unique node id, e.g. cam-1")
    ap.add_argument("--label", help="human name for the dashboard (default: the id)")
    ap.add_argument("--server", default="http://127.0.0.1:8800")
    ap.add_argument("--source", default="camera", choices=("camera", "synthetic"))
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--device", default="cpu",
                    help="cpu, or 0 for the GPU. Default cpu: a test node should not "
                         "compete for the card the federation is training on")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-fps", type=float, default=10.0, help="0 = as fast as it can")
    ap.add_argument("--frame-every", type=float, default=1.0,
                    help="seconds between picture uploads; telemetry is every frame")
    ap.add_argument("--frame-width", type=int, default=480)
    ap.add_argument("--check-every", type=int, default=50,
                    help="frames between checks for a newer global model")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
