"""Tests for the logic that can be wrong quietly.

The GPU stages are not unit-tested — `pipeline.verify` is what asserts those, and it
is the same four criteria the CI simulation-smoke job uses.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pipeline import gpu, logparse, paths, report, stages, vehicles
from pipeline.stages import Config

REPO = paths.REPO


# --------------------------------------------------------------- log parsing
CAPTURED = """
2026-08-05 - INFO - [Server] Aggregated parameters with checksum: -1032.5395936965942
2026-08-05 - INFO - [Server] Aggregated parameters with checksum: -2646.913425683975
2026-08-05 - INFO - [Client] Received weights with checksum: 698.6960870027542
2026-08-05 - INFO - [Client] Sending back weights with checksum: 679.3249893784523
2026-08-05 - INFO - [Client] Starting local training with batch_id=9, local_epochs=1
2026-08-05 - INFO - [Client] Initializing FlowerClient with model=models/yolov8s.pt on cuda:0
"""


def test_parses_negative_checksums():
    """Real aggregates are negative; a naive \\d+ pattern silently drops the sign."""
    vals = [e.value for e in logparse.parse_text(CAPTURED) if e.kind == "aggregate_checksum"]
    assert vals == [-1032.5395936965942, -2646.913425683975]


def test_detects_a_federation_that_did_not_learn(tmp_path):
    (tmp_path / "server.1.log").write_text(
        "Aggregated parameters with checksum: 647.578\n"
        "Aggregated parameters with checksum: 647.578\n")
    learned, detail = logparse.federation_learned(tmp_path)
    assert learned is False
    assert "did not change" in detail


def test_detects_a_federation_that_did_learn(tmp_path):
    (tmp_path / "server.1.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")
    assert logparse.federation_learned(tmp_path)[0] is True


def test_single_round_is_not_called_learning(tmp_path):
    (tmp_path / "server.1.log").write_text("Aggregated parameters with checksum: 1.0\n")
    learned, detail = logparse.federation_learned(tmp_path)
    assert learned is False and "need >=2" in detail


def test_no_optimizer_step_warning_is_parsed():
    line = ("[Client] batch=16 over 10 images x 1 epoch(s) gives 1 batch(es), "
            "fewer than the 4 needed for one optimizer step.")
    ev = logparse.parse_line(line)
    assert ev and ev.kind == "no_optimizer_step" and ev.value == 4


# ------------------------------------------------------------------ vehicles
def _index(n=400):
    return {f"img{i}.jpg": {"timeofday": "night" if i % 2 else "daytime",
                            "scene": "city street", "weather": "clear"} for i in range(n)}


def test_vehicles_get_condition_biased_slices():
    idx = _index()
    vs = vehicles.assign(2, 50, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=10, seed=1)
    assert [v.condition for v in vs] == ["daytime city", "night"]
    assert all(idx[n]["timeofday"] == "daytime" for n in vs[0].train)
    assert all(idx[n]["timeofday"] == "night" for n in vs[1].train)


def test_vehicle_slices_are_disjoint():
    """Overlap would train one image on two vehicles per round and flatter the aggregate."""
    idx = _index()
    vs = vehicles.assign(4, 30, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=5, seed=3)
    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen))


def test_vehicle_assignment_is_deterministic():
    idx = _index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=5, seed=7)
    assert [v.train for v in vehicles.assign(3, 20, **kw)] == \
           [v.train for v in vehicles.assign(3, 20, **kw)]


def test_short_condition_tops_up_rather_than_starving_a_vehicle():
    idx = {f"n{i}.jpg": {"timeofday": "night", "scene": "x", "weather": "clear"} for i in range(5)}
    idx.update({f"d{i}.jpg": {"timeofday": "daytime", "scene": "city street", "weather": "clear"}
                for i in range(100)})
    vs = vehicles.assign(2, 40, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=2, seed=0)
    night = next(v for v in vs if v.condition == "night")
    assert night.n_train == 40, "vehicle should be topped up, not left tiny"


# ----------------------------------------------------------------------- gpu
def test_energy_integration():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 50, 0, 16303, 100.0, 60))
    t.add(gpu.Sample(36.0, 50, 0, 16303, 100.0, 60))
    assert t.energy_wh == pytest.approx(1.0)


def test_energy_uses_the_midpoint_not_a_left_sum():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 0, 0, 16303, 0.0, 40))
    t.add(gpu.Sample(36.0, 0, 0, 16303, 200.0, 40))
    assert t.energy_wh == pytest.approx(1.0)   # a left-hand sum would give 0.0


def test_gpu_summary_reports_vram_against_the_measured_ceiling():
    t = gpu.Telemetry()
    t.add(gpu.Sample(0.0, 90, 15900, 16303, 250.0, 70))
    s = t.summary()
    assert s["peak_mem_mib"] == 15900 and 97 < s["peak_mem_pct"] < 98


# -------------------------------------------------------------------- stages
def test_every_expensive_stage_is_gated():
    """A browser tab must not be able to start a multi-hour GPU job unprompted."""
    gated = {s.name for s in stages.STAGES if s.gated}
    assert {"dataset", "sanity", "federate"} <= gated


def test_federate_runs_against_the_vehicle_root_not_my_project():
    fed = stages.BY_NAME["federate"]
    assert fed.data_root == paths.VEHICLE_ROOT


def test_snapshot_survives_a_broken_check(monkeypatch):
    boom = stages.Stage("boom", "Boom", False,
                        lambda c: (_ for _ in ()).throw(RuntimeError("nope")),
                        lambda c: ["true"])
    monkeypatch.setattr(stages, "STAGES", [boom])
    row = stages.snapshot(Config())[0]
    assert row["satisfied"] is False and "check failed" in row["detail"]


def test_profile_controls_size_and_resolution():
    assert Config(profile="demo").per_vehicle == 300
    assert Config(profile="demo").imgsz == 320
    assert Config(profile="full").per_vehicle == 6308
    assert Config(profile="full").imgsz == 640


# ------------------------------------------------------------------- reports
def _fixture_report():
    return {"generated": "now", "host": {"platform": "test", "python": "3.12"},
            "config": {"profile": "demo"}, "fleet": [{"vid": 1, "condition": "night",
                                                      "n_train": 300, "n_val": 60}],
            "per_vehicle": {}, "checksums": [1.0, 2.0], "learned": True,
            "learned_detail": "moved", "metrics": [{"round": 1, "stage": "evaluate", "mAP50": 0.3}],
            "gpu": {"energy_wh": 1.5, "peak_mem_mib": 15900, "peak_power_w": 250,
                    "mean_util_pct": 90}, "stages": [{"name": "federate", "status": "ok",
                                                      "seconds": 12.3}],
            "checkpoints": ["global_last.pt"]}


def test_report_renders_both_formats(tmp_path):
    h, m = report.write(_fixture_report(), tmp_path)
    assert h.exists() and m.exists()
    assert "run report" in h.read_text(encoding="utf-8")
    assert "| 1 | night | 300 | 60 |" in m.read_text(encoding="utf-8")


def test_report_html_is_self_contained(tmp_path):
    """No CDN, no external fetches -- it has to survive being emailed."""
    h, _ = report.write(_fixture_report(), tmp_path)
    text = h.read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', text)


def test_report_states_plainly_when_the_federation_did_not_learn(tmp_path):
    data = _fixture_report() | {"learned": False, "learned_detail": "identical"}
    h, m = report.write(data, tmp_path)
    assert "did NOT learn" in h.read_text(encoding="utf-8")
    assert "DID NOT LEARN" in m.read_text(encoding="utf-8")


# ----------------------------------------------------------------- isolation
def test_pipeline_never_writes_into_my_project():
    """The isolation guarantee, enforced rather than promised.

    Scans this package for writes whose target is my-project. The pipeline may only
    reach my-project by running its scripts as subprocesses.
    """
    offenders = []
    write_call = re.compile(r"\b(write_text|write_bytes|mkdir|unlink|rmtree|copy2|os\.link)\b")
    for py in sorted(Path(__file__).resolve().parents[1].glob("*.py")):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if write_call.search(line) and re.search(r"\bPROJECT\b", line):
                offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, "pipeline writes into my-project:\n" + "\n".join(offenders)


def test_vehicle_shards_live_outside_my_project():
    assert paths.PROJECT not in paths.VEHICLE_ROOT.parents
    assert paths.VEHICLE_ROOT.is_relative_to(paths.HERE)


def test_generated_paths_are_all_gitignored():
    """Dataset, checkpoints, MLflow store and reports must be uncommittable."""
    import subprocess
    targets = [paths.VEHICLE_ROOT / "batch" / "batch_1" / "images" / "x.jpg",
               paths.MLFLOW_STORE / "0" / "meta.yaml",
               paths.REPORTS / "20260805" / "report.html",
               paths.STATE / "attributes.json"]
    for t in targets:
        rel = t.relative_to(REPO).as_posix()
        out = subprocess.run(["git", "check-ignore", rel], cwd=REPO,
                             capture_output=True, text=True)
        assert out.returncode == 0, f"{rel} is NOT gitignored"
