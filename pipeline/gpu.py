"""GPU telemetry: utilisation, VRAM, power draw, and integrated energy.

Power is the number nobody logs and everybody asks about afterwards, so it is
sampled the whole time a run is in flight and integrated to watt-hours.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field

QUERY = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
# Measured on the RTX 5070 Ti: one client peaks at 15.9 GB of 16.3 GB, so the UI
# draws VRAM against a ceiling rather than a percentage nobody can interpret.
VRAM_CEILING_MIB = 16303


@dataclass
class Sample:
    t: float
    util_pct: float
    mem_used_mib: float
    mem_total_mib: float
    power_w: float
    temp_c: float


@dataclass
class Telemetry:
    """Rolling GPU stats plus energy integrated by the trapezoid rule."""

    samples: list[Sample] = field(default_factory=list)
    energy_wh: float = 0.0
    peak_mem_mib: float = 0.0
    peak_power_w: float = 0.0

    def add(self, s: Sample) -> None:
        if self.samples:
            prev = self.samples[-1]
            dt_h = (s.t - prev.t) / 3600.0
            # Trapezoid: power is a continuous signal sampled discretely; taking the
            # midpoint avoids the systematic bias a left-hand sum would accumulate
            # over a multi-hour run.
            self.energy_wh += (prev.power_w + s.power_w) / 2.0 * dt_h
        self.samples.append(s)
        self.peak_mem_mib = max(self.peak_mem_mib, s.mem_used_mib)
        self.peak_power_w = max(self.peak_power_w, s.power_w)

    @property
    def latest(self) -> Sample | None:
        return self.samples[-1] if self.samples else None

    def summary(self) -> dict:
        n = len(self.samples)
        return {
            "samples": n,
            "energy_wh": round(self.energy_wh, 4),
            "peak_mem_mib": round(self.peak_mem_mib, 1),
            "peak_mem_pct": round(100.0 * self.peak_mem_mib / VRAM_CEILING_MIB, 1),
            "peak_power_w": round(self.peak_power_w, 1),
            "mean_util_pct": round(sum(s.util_pct for s in self.samples) / n, 1) if n else 0.0,
        }


def read_once(timeout: float = 5.0) -> Sample | None:
    """One nvidia-smi sample, or None if there is no usable GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    try:
        # power.draw reads "[N/A]" on some laptop GPUs; treat it as zero rather than
        # failing the whole sampler over a field that is merely nice to have.
        def num(x: str) -> float:
            return float(x) if x.replace(".", "", 1).isdigit() else 0.0

        return Sample(time.time(), num(parts[0]), num(parts[1]), num(parts[2]), num(parts[3]), num(parts[4]))
    except (IndexError, ValueError):
        return None


class Sampler:
    """Background nvidia-smi poller. Start once, read `telemetry` any time."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.telemetry = Telemetry()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Sampler":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = read_once()
            if s:
                self.telemetry.add(s)
            self._stop.wait(self.interval)

    def stop(self) -> Telemetry:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
            self._thread = None
        return self.telemetry


def demo() -> None:
    """Self-check: energy integration against a hand-computable series."""
    t = Telemetry()
    # 100 W held for 36 s == 1 Wh.
    t.add(Sample(0.0, 50, 1000, 16303, 100.0, 60))
    t.add(Sample(36.0, 50, 1000, 16303, 100.0, 60))
    assert abs(t.energy_wh - 1.0) < 1e-9, t.energy_wh
    # Ramp 0 -> 200 W over 36 s averages 100 W, so also 1 Wh.
    t2 = Telemetry()
    t2.add(Sample(0.0, 0, 0, 16303, 0.0, 40))
    t2.add(Sample(36.0, 0, 0, 16303, 200.0, 40))
    assert abs(t2.energy_wh - 1.0) < 1e-9, t2.energy_wh
    assert t2.peak_power_w == 200.0
    print("gpu self-check OK", t.summary())


if __name__ == "__main__":
    demo()
