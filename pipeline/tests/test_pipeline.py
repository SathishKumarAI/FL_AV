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


# ------------------------------------------- exit codes that lie
def test_crash_in_output_is_detected_despite_exit_zero():
    """flwr exits 0 having printed 'Simulation Runtime crashed'. Believe the output."""
    lines = ["INFO: starting", "ERROR: Simulation Runtime crashed.", "ERROR: Exit Code: 700"]
    assert stages.scan_for_crash(lines, stages.CRASH_MARKERS) == "Simulation Runtime crashed"


def test_clean_output_is_not_flagged_as_a_crash():
    lines = ["INFO: Run finished 2 round(s) in 55.6s", "aggregate checksum: 1.0"]
    assert stages.scan_for_crash(lines, stages.CRASH_MARKERS) is None


def test_federate_stage_scans_for_crashes():
    assert stages.BY_NAME["federate"].crash_markers, "federate must not trust its exit code"


def test_init_args_are_omitted_when_attaching_to_an_existing_ray_cluster():
    """Ray: 'When connecting to an existing cluster, num_cpus and num_gpus must not
    be provided.' Passing them anyway crashes the simulation at startup."""
    attached = " ".join(stages._cmd_federate(Config(ray_address="127.0.0.1:6379")))
    standalone = " ".join(stages._cmd_federate(Config()))
    assert "init-args" not in attached
    assert "init-args-num-gpus=1" in standalone


def test_supernodes_track_the_vehicle_count():
    """Otherwise the run hangs forever waiting for clients that never arrive."""
    cmd = " ".join(stages._cmd_federate(Config(n_vehicles=6)))
    assert "num-supernodes=6" in cmd and "min_clients=6" in cmd


def test_fleet_check_demands_a_shard_for_every_assignable_id():
    """The server picks batch ids from DEFAULT_BATCH_ID_RANGE (1..10) at random and
    cannot be told to stay within the vehicle count, so every id must resolve."""
    import json as _json
    from pipeline import vehicles as _v
    original = _v.load_fleet
    try:
        _v.load_fleet = lambda: [{"vid": i, "condition": "x", "n_train": 300, "n_val": 60}
                                 for i in range(1, 7)]
        assert stages._check_fleet(Config(n_vehicles=6)).satisfied is False
        _v.load_fleet = lambda: [{"vid": i, "condition": "x", "n_train": 300, "n_val": 60}
                                 for i in range(1, 11)]
        assert stages._check_fleet(Config(n_vehicles=6)).satisfied is True
    finally:
        _v.load_fleet = original


# ------------------------------------------------------------------- server
def test_report_route_refuses_paths_outside_the_reports_dir():
    """`/reports/../../secret` must not escape. The check is `REPORTS in parents`."""
    from pipeline import server  # noqa: F401  (import guard: server must load cleanly)
    escaped = (paths.REPORTS / ".." / ".." / "CLAUDE.md").resolve()
    assert paths.REPORTS.resolve() not in escaped.parents

    inside = (paths.REPORTS / "20260101" / "report.html").resolve()
    assert paths.REPORTS.resolve() in inside.parents


def test_one_path_guard_covers_every_route_that_maps_a_url_to_a_file(tmp_path):
    """Reports and shard images share `safe_child`; both traversals must fail."""
    from pipeline import server

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "ok.jpg").write_bytes(b"x")
    (tmp_path / "secret.txt").write_text("credentials")

    assert server.safe_child(root, "sub/ok.jpg") == (root / "sub" / "ok.jpg").resolve()
    assert server.safe_child(root, "../secret.txt") is None
    assert server.safe_child(root, "..\\secret.txt") is None
    assert server.safe_child(root, "sub") is None            # a directory is not a file
    assert server.safe_child(root, "sub/missing.jpg") is None


def test_every_file_the_dashboard_imports_is_servable():
    """index.html references app.css and js/main.js; a 404 there renders a blank page."""
    from pipeline import server

    html = (server.STATIC / "index.html").read_text(encoding="utf-8")
    referenced = re.findall(r'(?:href|src)="/static/([^"]+)"', html)
    assert referenced, "index.html no longer references any static asset"
    for rel in referenced:
        target = server.safe_child(server.STATIC, rel)
        assert target is not None, f"{rel} is referenced but not servable"
        assert target.suffix in server.Handler.STATIC_TYPES, f"{rel} has no content type"

    # And every module those modules import, one level of transitive closure deep.
    for js in (server.STATIC / "js").glob("*.js"):
        for imported in re.findall(r'from "\./([^"]+)"', js.read_text(encoding="utf-8")):
            assert (server.STATIC / "js" / imported).is_file(), f"{js.name} imports missing {imported}"


def test_shard_composition_counts_what_the_vehicle_actually_holds(tmp_path, monkeypatch):
    """The condition label is a claim; this counts the images behind it."""
    batches = tmp_path / "batch"
    shard = batches / "batch_2"
    (shard / "images" / "train").mkdir(parents=True)
    (shard / "images" / "val").mkdir(parents=True)
    (shard / "train.txt").write_text("a.jpg\nb.jpg\nc.jpg\n")
    (shard / "images" / "train" / "a.jpg").write_bytes(b"x")
    (shard / "images" / "val" / "v.jpg").write_bytes(b"x")

    monkeypatch.setattr(paths, "VEHICLE_BATCHES", batches)
    index = {"a.jpg": {"weather": "rainy", "scene": "city street", "timeofday": "night"},
             "b.jpg": {"weather": "rainy", "scene": "highway", "timeofday": "night"}}
    comp = vehicles.composition(2, index=index)

    assert comp["n_train"] == 3 and comp["n_val"] == 1
    assert comp["counts"]["weather"] == {"rainy": 2, "unknown": 1}
    assert list(comp["counts"]["weather"]) == ["rainy", "unknown"]   # sorted by frequency
    assert comp["samples"] == ["a.jpg"]         # only the image that exists on disk


def test_composition_never_triggers_an_attribute_index_build(monkeypatch):
    """It runs inside a request handler; streaming 1.45 GB of JSON there hangs the page."""
    monkeypatch.setattr(vehicles, "build_attribute_index",
                        lambda *a, **k: pytest.fail("composition must not build the index"))
    monkeypatch.setattr(vehicles, "ATTR_CACHE", Path("does-not-exist.json"))
    monkeypatch.setattr(vehicles, "_ATTR_MEMO", None)
    assert vehicles.cached_attributes() == {}


def test_live_state_is_derived_from_disk_not_from_the_event_bus(monkeypatch, tmp_path):
    """A run launched from the CLI must still light up the dashboard."""
    from pipeline import server, verify as _verify

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server.1.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")
    (logs / "client.1.log").write_text(
        "Starting local training with batch_id=3, local_epochs=1\n"
        "Received weights with checksum: 10.0\n"
        "Sending back weights with checksum: 11.0\n")
    (logs / "metrics.csv").write_text(
        "round,stage,num_clients,loss,precision,recall,mAP50,mAP50-95,fitness\n"
        "1,evaluate,6,0.5,0.4,0.3,0.25,0.1,0.1\n")

    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    monkeypatch.setattr(_verify, "_metrics_csv", lambda: logs / "metrics.csv")

    live = server.State().live()
    assert live["rounds_done"] == 2
    assert live["checksums"] == [1.0, 2.0]
    assert live["map50"] == [0.25]
    assert live["per_vehicle"]["3"]["received"] == 10.0
    assert live["per_vehicle"]["3"]["sent"] == 11.0


# ------------------------------------------------ per-vehicle learning
from pipeline import vehicle_metrics as vm  # noqa: E402


def _client_log(tmp_path, body):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "client.1.log").write_text(body)
    return logs


def test_per_vehicle_metrics_are_parsed_without_eval(tmp_path, monkeypatch):
    """The dict comes from a log file. literal_eval, never eval."""
    body = ("[Client] 3 Training done. metrics={'precision': 0.30, 'recall': 0.27, "
            "'mAP50': 0.23, 'mAP50-95': 0.11, 'fitness': 0.11, 'os': 'Windows'}\n")
    logs = _client_log(tmp_path, body)
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    rounds = vm.per_vehicle_rounds()
    assert rounds["3"][0]["mAP50"] == 0.23
    assert rounds["3"][0]["mAP50_95"] == 0.11
    assert rounds["3"][0]["round"] == 1


def test_a_malicious_metrics_blob_cannot_execute(tmp_path, monkeypatch):
    logs = _client_log(tmp_path, "[Client] 1 Training done. metrics={'x': __import__('os')}\n")
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    assert vm.per_vehicle_rounds() == {}      # rejected, not executed


def test_divergence_sums_to_about_zero(tmp_path, monkeypatch):
    rounds = {"1": [{"round": 1, "mAP50": 0.20}],
              "2": [{"round": 1, "mAP50": 0.30}],
              "3": [{"round": 1, "mAP50": 0.40}]}
    div = vm.divergence(rounds)
    assert abs(sum(v[0] for v in div.values())) < 1e-9
    assert div["1"][0] < 0 < div["3"][0]


def test_a_vehicle_that_never_trained_does_not_break_aggregation(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "log_dirs", lambda: [tmp_path])
    assert vm.per_vehicle_rounds() == {}
    assert vm.divergence({}) == {}
    assert vm.weight_movement() == {}


def test_missing_results_csv_degrades_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT", tmp_path)
    assert vm.per_vehicle_epochs() == {}


def test_contribution_is_the_fedavg_weight():
    fleet = [{"vid": 1, "n_train": 300}, {"vid": 2, "n_train": 100}]
    c = vm.contribution(fleet)
    assert c["1"] == 0.75 and c["2"] == 0.25


def test_weight_movement_pairs_received_with_sent(tmp_path, monkeypatch):
    logs = _client_log(tmp_path,
        "Starting local training with batch_id=4, local_epochs=1\n"
        "Received weights with checksum: 100.0\n"
        "Sending back weights with checksum: 90.0\n")
    monkeypatch.setattr(paths, "log_dirs", lambda: [logs])
    assert vm.weight_movement()["4"] == [10.0]


def test_fleet_json_records_val_counts(tmp_path):
    """It recorded n_train only, so every report printed 'val | ?'."""
    v = vehicles.Vehicle(1, "night", ["a.jpg"] * 5, ["b.jpg"] * 2)
    assert v.to_summary() == {"vid": 1, "condition": "night", "n_train": 5, "n_val": 2}


# ------------------------------------------------------- partition strategies
def test_random_partition_ignores_conditions():
    """The IID control case: no condition filter, every vehicle a uniform draw."""
    idx = _index()
    vs = vehicles.assign(3, 30, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=5, seed=2, partition="random")
    assert {v.condition for v in vs} == {"random mix"}
    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen)), "random slices must still be disjoint"


def test_mixed_partition_alternates():
    idx = _index()
    vs = vehicles.assign(4, 20, index=idx, train_pool=set(idx), val_pool=set(idx),
                         val_per_vehicle=4, seed=5, partition="mixed")
    assert [v.condition for v in vs][1] == "random mix"
    assert [v.condition for v in vs][0] != "random mix"


def test_unknown_partition_is_rejected():
    idx = _index()
    with pytest.raises(ValueError):
        vehicles.assign(2, 10, index=idx, train_pool=set(idx), val_pool=set(idx),
                        partition="nonsense")


def test_fleet_check_catches_a_partition_mismatch(monkeypatch):
    """A condition fleet must not be silently reused for a random run."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "night", "n_train": 300, "n_val": 60} for i in range(1, 11)])
    assert stages._check_fleet(Config(n_vehicles=6, partition="random")).satisfied is False
    assert stages._check_fleet(Config(n_vehicles=6, partition="condition")).satisfied is True
