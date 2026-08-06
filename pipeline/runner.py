"""Run the pipeline. Usable from the CLI or driven by the control dashboard.

    python -m pipeline.runner --list
    python -m pipeline.runner --stages env,fleet
    python -m pipeline.runner --all --profile demo --vehicles 6 --yes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import gpu, logparse, paths, stages, vehicles
from .stages import Config, Stage


@dataclass
class StageResult:
    name: str
    status: str                 # skipped | ok | failed | aborted
    seconds: float = 0.0
    detail: str = ""
    tail: list[str] = field(default_factory=list)
    gpu: dict = field(default_factory=dict)


class Run:
    """One pipeline invocation. Emits events; the dashboards subscribe to them."""

    def __init__(self, cfg: Config, confirm_all: bool = False, ray_address: str | None = None):
        self.cfg = cfg
        self.confirm_all = confirm_all
        self.ray_address = ray_address
        self.results: list[StageResult] = []
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.sampler = gpu.Sampler()
        self.started = time.time()
        self.current: str | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    # -- event plumbing ----------------------------------------------------
    def emit(self, kind: str, **payload) -> None:
        self.events.put({"kind": kind, "t": time.time(), **payload})

    def _emit_log(self, stage: str, line: str) -> None:
        self.emit("log", stage=stage, line=line)
        ev = logparse.parse_line(line)
        if ev:
            # Structured facts (checksums, which vehicle is training, no-op warnings)
            # ride the same stream so the live view never has to re-parse raw text.
            self.emit("signal", stage=stage, signal=ev.kind, value=ev.value, extra=ev.extra or {})

    # -- execution ---------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run_stage(self, stage: Stage) -> StageResult:
        check = stage.check(self.cfg)
        if check.satisfied:
            self.emit("stage", stage=stage.name, status="skipped", detail=check.detail)
            return StageResult(stage.name, "skipped", detail=check.detail)

        if stage.gated and not self.confirm_all:
            self.emit("stage", stage=stage.name, status="needs_confirm", detail=stage.est)
            return StageResult(stage.name, "aborted", detail=f"needs confirmation ({stage.est})")

        try:
            cmd = stage.command(self.cfg)
        except Exception as e:
            self.emit("stage", stage=stage.name, status="failed", detail=str(e))
            return StageResult(stage.name, "failed", detail=str(e))

        if stage.name == "federate":
            self._stop_stale_superlink()
        env = paths.subprocess_env(self.ray_address, data_root=stage.data_root)
        self.current = stage.name
        self.emit("stage", stage=stage.name, status="running", detail=" ".join(map(str, cmd))[:200])
        t0 = time.time()
        before_wh = self.sampler.telemetry.energy_wh

        tail: list[str] = []
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=stage.cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
            )
            for line in self._proc.stdout:            # stream, do not buffer to the end
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-200]                       # keep the last 200 lines only
                self._emit_log(stage.name, line)
            code = self._proc.wait()
        except Exception as e:
            self.emit("stage", stage=stage.name, status="failed", detail=str(e))
            return StageResult(stage.name, "failed", time.time() - t0, str(e), tail)
        finally:
            self._proc = None
            self.current = None

        secs = time.time() - t0
        gpu_used = {"energy_wh": round(self.sampler.telemetry.energy_wh - before_wh, 4),
                    "peak_mem_mib": self.sampler.telemetry.peak_mem_mib}

        # Exit code is necessary but not sufficient: flwr returns 0 having printed
        # "Simulation Runtime crashed". Believe the output over the status code.
        crash = stages.scan_for_crash(tail, stage.crash_markers)
        if code == 0 and crash:
            status, detail = "failed", f"exit 0 but output says: {crash}"
        else:
            status, detail = ("ok" if code == 0 else "failed"), f"exit {code}"

        self.emit("stage", stage=stage.name, status=status, detail=detail,
                  seconds=round(secs, 1), gpu=gpu_used)
        return StageResult(stage.name, status, secs, detail, tail, gpu_used)

    def execute(self, chain: list[Stage]) -> bool:
        self.sampler.start()
        self.emit("run_start", config=self.cfg.to_dict(), stages=[s.name for s in chain])
        ok = True
        try:
            for stage in chain:
                if self._stop.is_set():
                    self.results.append(StageResult(stage.name, "aborted", detail="stopped"))
                    break
                res = self.run_stage(stage)
                self.results.append(res)
                if res.status in ("failed", "aborted"):
                    # Halt: continuing past a failure is how silent no-ops get shipped.
                    ok = False
                    self.emit("run_halt", stage=stage.name, reason=res.detail)
                    break
        finally:
            telemetry = self.sampler.stop()
            self._restore_pyproject()
            report_paths = self._write_report(telemetry.summary())
            self.emit("run_end", ok=ok, seconds=round(time.time() - self.started, 1),
                      gpu=telemetry.summary(),
                      report=[str(p) for p in report_paths],
                      results=[r.__dict__ for r in self.results])
        return ok

    def _write_report(self, telemetry: dict) -> list[Path]:
        """Always write the report, including for a failed run -- that is when the
        inputs and the partial timeline are most worth having."""
        try:
            from . import report
            data = report.collect(config=self.cfg.to_dict(), telemetry=telemetry,
                                  results=[r.__dict__ for r in self.results])
            return list(report.write(data))
        except Exception as e:
            # Reporting must not fail the run -- but it must not be silent either.
            # It failed once and the only trace was a log line that a filtered console
            # dropped, so the run looked clean and produced nothing.
            import traceback
            detail = f"report generation FAILED: {type(e).__name__}: {e}"
            print(f"\n!! {detail}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.results.append(StageResult("report", "failed", detail=detail))
            self.emit("stage", stage="report", status="failed", detail=detail)
            return []

    def _stop_stale_superlink(self) -> None:
        """Kill any running flower-superlink before starting a federation.

        The SuperLink is detached and long-lived: it caches the working directory AND
        the environment of whichever `flwr run` first started it. A run that later
        changes RAY_ADDRESS or the data root inherits the old ones, and the simulation
        dies with an error that points at Ray rather than at the real cause. Starting
        from a clean SuperLink is cheaper than reasoning about which env it holds.
        """
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/IM", "flower-superlink.exe"],
                               capture_output=True, timeout=30)
            else:
                subprocess.run(["pkill", "-f", "flower-superlink"],
                               capture_output=True, timeout=30)
            self.emit("log", stage="federate", line="stopped any stale flower-superlink")
        except (OSError, subprocess.SubprocessError) as e:
            self.emit("log", stage="federate", line=f"could not stop superlink: {e}")

    @staticmethod
    def _restore_pyproject() -> None:
        """`flwr run` comments out [tool.flwr.federations] in place. Put it back.

        Committing the rewritten file leaves a fresh clone with no federation to run,
        which has already happened once in this repo's history.
        """
        pp = paths.PROJECT / "pyproject.toml"
        if pp.exists() and "CONFIGURATION MIGRATION NOTICE" in pp.read_text(errors="replace"):
            if shutil.which("git"):
                subprocess.run(["git", "checkout", "--", str(pp)], cwd=paths.REPO,
                               capture_output=True)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show stages and their current state")
    ap.add_argument("--stages", help="comma-separated subset, e.g. env,fleet")
    ap.add_argument("--all", action="store_true", help="run the whole chain")
    ap.add_argument("--profile", choices=("demo", "full"), default="demo")
    ap.add_argument("--vehicles", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-vehicle", type=int, default=0,
                    help="override images per vehicle (0 = profile default)")
    ap.add_argument("--partition", default="condition", choices=vehicles.PARTITIONS,
                    help="condition = non-IID (default); random = IID control; "
                         "mixed = both; dirichlet = tunable skew via --alpha")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dirichlet concentration: 0.05 concentrates each vehicle on one "
                         "condition, 100 is effectively IID")
    ap.add_argument("--yes", action="store_true", help="confirm the gated stages up front")
    ap.add_argument("--ray-address", help="attach to an existing Ray head (enables its dashboard)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(profile=args.profile, n_vehicles=args.vehicles,
                 rounds=args.rounds, local_epochs=args.epochs, seed=args.seed,
                 partition=args.partition, alpha=args.alpha,
                 per_vehicle_override=args.per_vehicle,
                 ray_address=args.ray_address)

    if args.list or not (args.stages or args.all):
        rows = stages.snapshot(cfg)
        if args.json:
            print(json.dumps({"config": cfg.to_dict(), "stages": rows}, indent=1))
            return 0
        print(f"profile={cfg.profile} vehicles={cfg.n_vehicles} rounds={cfg.rounds} "
              f"epochs={cfg.local_epochs} images/vehicle={cfg.per_vehicle} imgsz={cfg.imgsz}\n")
        for r in rows:
            mark = "skip" if r["satisfied"] else ("GATE" if r["gated"] else "run ")
            print(f"  [{mark}] {r['name']:<9} {r['title']:<26} {r['est']:<18} {r['detail']}")
        print("\nnothing was run; pass --stages <names> or --all")
        return 0

    chain = stages.resolve(args.stages) if args.stages else list(stages.STAGES)
    run = Run(cfg, confirm_all=args.yes, ray_address=args.ray_address)

    printer = threading.Thread(target=_drain, args=(run,), daemon=True)
    printer.start()
    ok = run.execute(chain)
    time.sleep(0.2)   # let the drain flush the tail of the queue
    return 0 if ok else 1


def _drain(run: Run) -> None:
    while True:
        ev = run.events.get()
        kind = ev.get("kind")
        if kind == "log":
            print(f"    {ev['line']}")
        elif kind == "stage":
            print(f"[{ev['status']:>13}] {ev['stage']:<9} {ev.get('detail','')}")
        elif kind == "run_halt":
            print(f"HALTED at {ev['stage']}: {ev['reason']}")
        elif kind == "run_end":
            g = ev.get("gpu", {})
            print(f"\nrun {'OK' if ev['ok'] else 'FAILED'} in {ev['seconds']}s | "
                  f"GPU energy {g.get('energy_wh', 0)} Wh, peak VRAM {g.get('peak_mem_mib', 0)} MiB")
            for p in ev.get("report", []):
                print(f"report: {p}")
            return


if __name__ == "__main__":
    raise SystemExit(main())
