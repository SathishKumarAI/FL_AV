"""The two dashboards: a control view to launch runs, a live view to watch them.

Scope is deliberately narrow. MLflow owns metrics storage, history and run
comparison; the Ray Dashboard owns actor and GPU internals. This server only does
what neither can: start a run from a form, and narrate a fleet of vehicles while it
trains. It stores nothing — every number here is read from a running subprocess,
from my-project's logs, or from nvidia-smi.

Stdlib only, loopback only, no build step.

    python -m pipeline.server           # then open http://127.0.0.1:8800
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import gpu, logparse, paths, stages, vehicle_metrics, vehicles, verify
from .runner import Run
from .stages import Config

STATIC = Path(__file__).resolve().parent / "static"
HISTORY_LIMIT = 500


def safe_child(root: Path, rel: str) -> Path | None:
    """``root/rel`` if it is a real file genuinely under ``root``, else None.

    One guard for every route that maps a URL onto a path. Written once because two
    copies of a traversal check is one copy too many: `/reports/../../secret` and
    `/api/shard-image/1/../../../secret` are the same bug.
    """
    target = (root / rel).resolve()
    if not target.is_file() or root.resolve() not in target.parents:
        return None
    return target


class Broadcaster:
    """Fan one Run's event queue out to every connected browser."""

    def __init__(self):
        self.subscribers: list["list"] = []
        self.history: list[dict] = []
        self.lock = threading.Lock()

    def publish(self, ev: dict) -> None:
        with self.lock:
            self.history.append(ev)
            del self.history[:-HISTORY_LIMIT]
            for q in self.subscribers:
                q.append(ev)

    def subscribe(self) -> list:
        q: list = []
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: list) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


class State:
    """Everything the dashboards need, and the one run allowed at a time."""

    def __init__(self):
        self.bus = Broadcaster()
        self.run: Run | None = None
        self.thread: threading.Thread | None = None
        self.idle_sampler = gpu.Sampler(interval=3.0).start()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: Config, chain, confirm: bool, ray_address: str | None) -> bool:
        if self.busy:
            return False
        run = Run(cfg, confirm_all=confirm, ray_address=ray_address)
        self.run = run

        def pump():
            while True:
                ev = run.events.get()
                self.bus.publish(ev)
                if ev.get("kind") == "run_end":
                    return

        threading.Thread(target=pump, daemon=True).start()
        self.thread = threading.Thread(target=run.execute, args=(chain,), daemon=True)
        self.thread.start()
        return True

    def live(self) -> dict:
        """Run state read from disk.

        Derived from logs and metrics rather than from this server's event bus, so a
        run launched from the CLI -- or one that started before this server did --
        still shows up. The event bus stays for the low-latency log stream.
        """
        checksums = logparse.aggregate_checksums()
        metrics_path = verify._metrics_csv()
        rows = logparse.read_metrics_csv(metrics_path)

        per_vehicle: dict[str, dict] = {}
        current = None
        noop = 0
        for f in logparse.iter_logs("client*.log"):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for ev in logparse.parse_text(text):
                if ev.kind == "training_start":
                    current = str(ev.value)
                    v = per_vehicle.setdefault(current, {"rounds": 0, "received": None,
                                                         "sent": None, "device": None})
                    v["rounds"] += 1
                elif ev.kind == "no_optimizer_step":
                    noop += 1
                elif current:
                    v = per_vehicle.setdefault(current, {"rounds": 0, "received": None,
                                                         "sent": None, "device": None})
                    if ev.kind == "client_received_checksum":
                        v["received"] = ev.value
                    elif ev.kind == "client_sent_checksum":
                        v["sent"] = ev.value
                    elif ev.kind == "device":
                        v["device"] = ev.value

        ok, criteria = verify.check()
        evaluated = [r for r in rows if r.get("stage") == "evaluate"]
        return {
            "checksums": checksums,
            "rounds_done": len(checksums),
            "metrics": rows,
            "map50": [r.get("mAP50") for r in evaluated],
            "loss": [r.get("loss") for r in evaluated],
            "per_vehicle": per_vehicle,
            "training_now": current if self.busy else None,
            "no_optimizer_steps": noop,
            "criteria": criteria,
            "criteria_ok": ok,
            "checkpoints": sorted(p.name for p in (paths.PROJECT / "checkpoints").glob("global_*.pt")),
            "learning": vehicle_metrics.summary(),
        }

    def reports(self) -> list[dict]:
        out = []
        if paths.REPORTS.is_dir():
            for d in sorted(paths.REPORTS.iterdir(), reverse=True)[:12]:
                if (d / "report.html").exists():
                    out.append({"name": d.name, "url": f"/reports/{d.name}/report.html"})
        return out

    def snapshot(self, cfg: Config) -> dict:
        live = self.run.sampler.telemetry if self.busy and self.run else self.idle_sampler.telemetry
        latest = live.latest
        return {
            "busy": self.busy,
            "current": self.run.current if self.run else None,
            "config": cfg.to_dict(),
            "stages": stages.snapshot(cfg),
            "fleet": vehicles.load_fleet(),
            "gpu": {
                "util_pct": latest.util_pct if latest else None,
                "mem_used_mib": latest.mem_used_mib if latest else None,
                "mem_ceiling_mib": gpu.VRAM_CEILING_MIB,
                "power_w": latest.power_w if latest else None,
                "temp_c": latest.temp_c if latest else None,
                "history": [{"util": s.util_pct, "power": s.power_w, "mem": s.mem_used_mib}
                            for s in live.samples[-120:]],
                **live.summary(),
            },
            "links": {"mlflow": "http://127.0.0.1:5000", "ray": "http://127.0.0.1:8265"},
            "results": [r.__dict__ for r in (self.run.results if self.run else [])],
            "live": self.live(),
            "reports": self.reports(),
        }


STATE = State()
CONFIG = Config()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):        # the browser polls; the console stays readable
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._json(STATE.snapshot(CONFIG))
        if self.path == "/api/events":
            return self._sse()
        if self.path.startswith("/static/"):
            return self._static()
        if self.path.startswith("/api/vehicle/"):
            return self._vehicle()
        if self.path.startswith("/api/shard-image/"):
            return self._shard_image()
        if self.path.startswith("/reports/"):
            return self._report_file()
        self._json({"error": "not found"}, 404)

    #: Only what the dashboard is made of. An unknown suffix is a 404, not an
    #: octet-stream download, so a stray file here cannot be exfiltrated by URL.
    STATIC_TYPES = {
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def _static(self) -> None:
        """Serve the dashboard's own CSS and JS modules, straight off disk.

        Read per request on purpose: an edit is live on reload, which is what makes
        a no-build-step page worth having.
        """
        target = safe_child(STATIC, self.path[len("/static/"):].split("?")[0])
        if target is None or target.suffix not in self.STATIC_TYPES:
            return self._json({"error": "not found"}, 404)
        self._send(200, target.read_bytes(), self.STATIC_TYPES[target.suffix])

    def _vehicle(self) -> None:
        """Shard composition for one vehicle, for the detail drawer."""
        try:
            vid = int(self.path.rsplit("/", 1)[-1].split("?")[0])
        except ValueError:
            return self._json({"error": "vehicle id must be an integer"}, 400)
        self._json(vehicles.composition(vid))

    def _shard_image(self) -> None:
        """One image out of one vehicle's shard: what its condition looks like."""
        rel = self.path[len("/api/shard-image/"):].split("?")[0]
        vid, _, name = rel.partition("/")
        if not vid.isdigit():
            return self._json({"error": "vehicle id must be an integer"}, 400)
        root = paths.VEHICLE_BATCHES / f"batch_{int(vid)}" / "images" / "train"
        target = safe_child(root, name)
        if target is None:
            return self._json({"error": "not found"}, 404)
        ctype = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
        self._send(200, target.read_bytes(), ctype)

    def _report_file(self) -> None:
        """Serve a generated report, refusing anything outside the reports dir."""
        rel = self.path[len("/reports/"):].split("?")[0]
        target = safe_child(paths.REPORTS, rel)
        if target is None:
            return self._json({"error": "not found"}, 404)
        ctype = "text/html; charset=utf-8" if target.suffix == ".html" else "text/plain; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = STATE.bus.subscribe()
        try:
            for ev in list(STATE.bus.history):     # replay so a late tab is not blank
                self._event(ev)
            while True:
                if q:
                    self._event(q.pop(0))
                else:
                    # Comment frame doubles as a keep-alive and as the signal that
                    # tells us the browser has gone away (raises on write).
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            STATE.bus.unsubscribe(q)

    def _event(self, ev: dict) -> None:
        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/run":
            global CONFIG
            CONFIG = Config(
                profile=body.get("profile", "demo"),
                n_vehicles=int(body.get("vehicles", 6)),
                rounds=int(body.get("rounds", 2)),
                local_epochs=int(body.get("epochs", 1)),
                seed=int(body.get("seed", 0)),
                partition=body.get("partition", "condition"),
                alpha=float(body.get("alpha", 0.5) or 0.5),
                ray_address=body.get("ray_address") or None,
            )
            if CONFIG.partition not in vehicles.PARTITIONS:
                # Rejected here rather than three subprocesses later, where it would
                # surface as a stage failure with the real cause buried in a log.
                return self._json({"error": f"unknown partition {CONFIG.partition!r}; "
                                            f"known: {', '.join(vehicles.PARTITIONS)}"}, 400)
            try:
                chain = stages.resolve(body.get("stages"))
            except SystemExit as e:
                return self._json({"error": str(e)}, 400)
            started = STATE.start(CONFIG, chain, bool(body.get("confirm")),
                                  body.get("ray_address") or None)
            return self._json({"started": started, "busy": STATE.busy},
                              200 if started else 409)

        if self.path == "/api/stop":
            if STATE.run:
                STATE.run.stop()
            return self._json({"stopping": True})

        self._json({"error": "not found"}, 404)


def serve(port: int = 8800) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # loopback only, never 0.0.0.0
    print(f"control + live dashboards : http://127.0.0.1:{port}")
    print(f"MLflow (metrics, history) : http://127.0.0.1:5000   [mlflow ui --port 5000]")
    print(f"Ray (actors, GPU internals): http://127.0.0.1:8265  [ray start --head]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8800)
    serve(ap.parse_args().port)
