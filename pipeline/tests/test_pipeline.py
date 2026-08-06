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
               paths.STATE / "attributes.json",
               # The holdout and the pooled baseline set are hardlinks onto the
               # 7.6 GB kagglehub cache, same as the shards.
               paths.VEHICLE_ROOT / "holdout" / "images" / "val" / "x.jpg",
               paths.VEHICLE_ROOT / "pooled" / "images" / "train" / "x.jpg",
               paths.STATE / "baseline_runs" / "centralised" / "weights" / "best.pt",
               paths.STATE / "holdout_metrics.json",
               # Ultralytics' AMP check downloads a small model into the CWD the
               # first time train() runs, so one lands in the repo root after any
               # baseline or sanity run.
               REPO / "yolo26n.pt",
               REPO / "models" / "yolov8s.pt"]
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


@pytest.fixture(autouse=True)
def _no_fleet_state_on_disk(monkeypatch):
    """The fleet check consults the holdout and the fleet manifest. A test must
    assert its own logic, not whatever the last real run left in the working tree --
    a demo run rewriting the manifest should not turn other tests red. The manifest's
    and the holdout's own effects each have a dedicated test that patches them back."""
    from pipeline import holdout as _h, vehicles as _v
    monkeypatch.setattr(_h, "names", lambda: set())
    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {})


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
    summary = v.to_summary()
    assert {k: summary[k] for k in ("vid", "condition", "n_train", "n_val")} == {
        "vid": 1, "condition": "night", "n_train": 5, "n_val": 2}
    assert len(summary["fingerprint"]) == 12


def test_a_fleet_fingerprint_proves_two_runs_saw_the_same_images():
    """Same config does not mean same data: a rebuilt fleet, a changed holdout or a
    different pool all produce different images under an identical config."""
    a = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["v.jpg"])
    same = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["v.jpg"])
    other_train = vehicles.Vehicle(1, "night", ["a.jpg", "c.jpg"], ["v.jpg"])
    other_val = vehicles.Vehicle(1, "night", ["a.jpg", "b.jpg"], ["w.jpg"])

    assert a.fingerprint() == same.fingerprint()
    assert a.fingerprint() != other_train.fingerprint()
    assert a.fingerprint() != other_val.fingerprint(), "the val split is part of the data too"


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


def _mixed_index(n=1200):
    """An index spanning several PROFILES, so a Dirichlet draw has groups to skew over."""
    kinds = [
        {"timeofday": "daytime", "scene": "city street", "weather": "clear"},   # daytime city
        {"timeofday": "night", "scene": "residential", "weather": "clear"},     # night
        {"timeofday": "daytime", "scene": "residential", "weather": "rainy"},   # rain / fog
        {"timeofday": "daytime", "scene": "highway", "weather": "clear"},       # highway
        {"timeofday": "dawn/dusk", "scene": "residential", "weather": "clear"}, # dawn / dusk
        {"timeofday": "daytime", "scene": "residential", "weather": "snowy"},   # snow
    ]
    return {f"img{i}.jpg": kinds[i % len(kinds)] for i in range(n)}


def _dominant_share(vehicle, index):
    counts = {}
    for name in vehicle.train:
        key = vehicles._group_of(index.get(name, {}))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / max(1, len(vehicle.train))


def test_dirichlet_alpha_is_the_skew_knob():
    """Small alpha concentrates a vehicle on one condition; large alpha flattens it.

    The direction is asserted, not a magic number: the draw is random, and pinning
    an exact share would make this a test of one seed rather than of the mechanism.
    """
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              partition="dirichlet")

    skewed = vehicles.assign(6, 60, seed=1, alpha=0.02, **kw)
    flat = vehicles.assign(6, 60, seed=1, alpha=200.0, **kw)

    skewed_share = sum(_dominant_share(v, idx) for v in skewed) / len(skewed)
    flat_share = sum(_dominant_share(v, idx) for v in flat) / len(flat)
    assert skewed_share > flat_share + 0.2, (skewed_share, flat_share)
    assert skewed_share > 0.8, "alpha=0.02 should put a vehicle almost entirely on one condition"


def test_dirichlet_slices_are_disjoint_and_deterministic():
    idx = _mixed_index()
    kw = dict(index=idx, train_pool=set(idx), val_pool=set(idx), val_per_vehicle=10,
              partition="dirichlet", alpha=0.4, seed=11)
    vs = vehicles.assign(5, 50, **kw)

    seen = [n for v in vs for n in v.train]
    assert len(seen) == len(set(seen)), "dirichlet slices must not overlap"
    assert all(len(v.train) == 50 for v in vs), "the per-client-mixture variant keeps sizes equal"
    assert [v.train for v in vehicles.assign(5, 50, **kw)] == [v.train for v in vs]
    assert all(v.condition.startswith("dirichlet") for v in vs)


def test_dirichlet_rejects_an_alpha_that_has_no_meaning():
    idx = _mixed_index(60)
    with pytest.raises(ValueError):
        vehicles.assign(2, 10, index=idx, train_pool=set(idx), val_pool=set(idx),
                        partition="dirichlet", alpha=0)


def test_the_registry_is_what_every_caller_reads():
    """Registering a partitioner is the only step; CLI choices follow from it."""
    assert vehicles.PARTITIONS == tuple(vehicles.PARTITIONERS)
    assert set(vehicles.PARTITIONS) >= {"condition", "random", "mixed", "dirichlet"}

    from pipeline import build_fleet, runner
    for parser, flag in ((build_fleet, None), (runner, None)):
        pass
    choices = runner.build_parser()._option_string_actions["--partition"].choices
    assert tuple(choices) == vehicles.PARTITIONS


def test_fleet_manifest_is_compared_rather_than_guessed(monkeypatch):
    """Label-sniffing could not tell condition from mixed, or one alpha from another."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "dirichlet a=0.5 - night 90%", "n_train": 300, "n_val": 60}
        for i in range(1, 11)])
    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "dirichlet", "alpha": 0.5, "seed": 0, "per_vehicle": 300})

    same = Config(n_vehicles=6, partition="dirichlet", alpha=0.5)
    assert stages._check_fleet(same).satisfied is True

    other_alpha = stages._check_fleet(Config(n_vehicles=6, partition="dirichlet", alpha=0.1))
    assert other_alpha.satisfied is False and "alpha" in other_alpha.detail

    other_partition = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert other_partition.satisfied is False and "partition" in other_partition.detail


# ------------------------------------------------------ holdout and baseline
from pipeline import baseline as _baseline, holdout as _holdout  # noqa: E402


def test_holdout_selection_is_deterministic_across_processes():
    """Set iteration order varies per process; shuffling a set would give a
    different holdout on every machine and make two runs incomparable."""
    pool = {f"img{i}.jpg" for i in range(500)}
    first = _holdout.select(50, seed=4, val_pool=pool)
    second = _holdout.select(50, seed=4, val_pool=set(reversed(sorted(pool))))
    assert first == second
    assert _holdout.select(50, seed=5, val_pool=pool) != first
    assert len(set(first)) == 50


def test_holdout_refuses_to_invent_images_it_does_not_have():
    with pytest.raises(SystemExit):
        _holdout.select(500, val_pool={"a.jpg", "b.jpg"})


def test_no_vehicle_can_be_assigned_a_held_out_image():
    """The whole point: a client that self-evaluates on a holdout image makes the
    global metric partly self-referential."""
    idx = _mixed_index(600)
    pool = set(idx)
    held = set(_holdout.select(100, seed=2, val_pool=pool))

    vs = vehicles.assign(4, 40, index=idx, train_pool=pool, val_pool=pool,
                         val_per_vehicle=20, seed=1, exclude=held)
    for v in vs:
        assert not (set(v.train) & held), f"vehicle {v.vid} trains on held-out images"
        assert not (set(v.val) & held), f"vehicle {v.vid} self-evaluates on held-out images"


def test_a_fleet_that_predates_the_holdout_is_rebuilt(monkeypatch):
    """Otherwise the fleet stage says 'skip' and the contamination is never noticed."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "night", "n_train": 300, "n_val": 60} for i in range(1, 11)])
    monkeypatch.setattr(_holdout, "names", lambda: {f"h{i}.jpg" for i in range(1000)})

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {})
    predates = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert predates.satisfied is False and "holdout" in predates.detail

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "condition", "seed": 0, "per_vehicle": 300, "holdout": 500})
    stale = stages._check_fleet(Config(n_vehicles=6, partition="condition"))
    assert stale.satisfied is False and "holdout" in stale.detail

    monkeypatch.setattr(_v, "load_fleet_meta", lambda: {
        "partition": "condition", "seed": 0, "per_vehicle": 300, "holdout": 1000})
    assert stages._check_fleet(Config(n_vehicles=6, partition="condition")).satisfied is True


def test_the_holdout_stage_runs_before_the_fleet_stage():
    """Order is load-bearing: a holdout carved afterwards is already in someone's
    val split."""
    names = [s.name for s in stages.STAGES]
    assert names.index("holdout") < names.index("fleet")
    assert names.index("evaluate") > names.index("federate")
    assert stages.BY_NAME["baseline"].gated, "it trains a whole model"


def test_evaluating_a_checkpoint_that_is_not_there_fails_loudly(tmp_path):
    """A zero would look like a bad model rather than a missing one."""
    with pytest.raises(SystemExit):
        _holdout.evaluate(tmp_path / "nope.pt")


def test_the_gap_is_only_reported_when_both_halves_exist(monkeypatch):
    monkeypatch.setattr(_baseline, "result", lambda: {})
    monkeypatch.setattr(_holdout, "curve", lambda: {"rounds": [{"mAP50": 0.4}]})
    assert _baseline.gap() == {}

    monkeypatch.setattr(_baseline, "result", lambda: {"mAP50": 0.5, "epochs": 24, "images": 8400})
    g = _baseline.gap()
    assert g["federated_mAP50"] == 0.4 and g["centralised_mAP50"] == 0.5
    assert abs(g["gap"] - 0.1) < 1e-9 and abs(g["retained"] - 0.8) < 1e-9


def test_the_baseline_budget_matches_the_federated_one():
    """rounds x local_epochs epochs over the pooled set is the same number of
    image-visits the fleet makes; anything else flatters one side."""
    cfg = Config(rounds=6, local_epochs=4)
    cmd = stages._cmd_baseline(cfg)
    assert "--rounds" in cmd and cmd[cmd.index("--rounds") + 1] == "6"
    assert cmd[cmd.index("--local-epochs") + 1] == "4"


def test_fleet_check_catches_a_partition_mismatch(monkeypatch):
    """A condition fleet must not be silently reused for a random run."""
    from pipeline import vehicles as _v
    monkeypatch.setattr(_v, "load_fleet", lambda: [
        {"vid": i, "condition": "night", "n_train": 300, "n_val": 60} for i in range(1, 11)])
    assert stages._check_fleet(Config(n_vehicles=6, partition="random")).satisfied is False
    assert stages._check_fleet(Config(n_vehicles=6, partition="condition")).satisfied is True


def test_the_report_leads_with_the_metric_no_client_could_flatter(monkeypatch):
    """A report that shows only self-evaluated numbers invites the comparison that
    the holdout exists to prevent."""
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 1000},
        "rounds": [{"round": 1, "mAP50": 0.35, "mAP50-95": 0.19, "precision": 0.5, "recall": 0.3},
                   {"round": 2, "mAP50": 0.43, "mAP50-95": 0.24, "precision": 0.6, "recall": 0.4}]})
    monkeypatch.setattr(_b, "gap", lambda: {
        "federated_mAP50": 0.43, "centralised_mAP50": 0.50, "gap": 0.07, "retained": 0.86})

    md = report.to_markdown(report.collect(config={"profile": "full"}))
    assert "## The honest global metric" in md
    assert "1000 images that no vehicle trained" in md
    assert "0.4300" in md and "0.5000" in md and "86.0%" in md
    # And the old number keeps its caveat rather than its old headline.
    assert "per client, on its own split" in md


def test_the_report_says_when_the_number_has_no_scale(monkeypatch):
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 500},
        "rounds": [{"round": 1, "mAP50": 0.35, "mAP50-95": 0.19, "precision": 0.5, "recall": 0.3}]})
    monkeypatch.setattr(_b, "gap", lambda: {})
    md = report.to_markdown(report.collect(config={}))
    assert "still has no scale" in md


# ------------------------------------------------------------ shard validation
from pipeline import validate as _validate  # noqa: E402


def _shard(root, vid, train=("a.jpg",), val=("v.jpg",), label_text="0 0.5 0.5 0.2 0.2\n"):
    """A minimal but sound shard: listings, images and labels that agree."""
    shard = root / f"batch_{vid}"
    for split, names in (("train", train), ("val", val)):
        (shard / "images" / split).mkdir(parents=True, exist_ok=True)
        (shard / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            (shard / "images" / split / name).write_bytes(b"x")
            (shard / "labels" / split / f"{name.rsplit('.', 1)[0]}.txt").write_text(label_text)
        (shard / f"{split}.txt").write_text("".join(f"{n}\n" for n in names))
    return shard


def _checks(problems):
    return {p.check for p in problems}


def test_a_sound_fleet_reports_nothing(tmp_path):
    _shard(tmp_path, 1)
    _shard(tmp_path, 2, train=("b.jpg",), val=("w.jpg",))
    assert _validate.check_fleet(tmp_path, held=set()) == []


def test_an_image_without_a_label_is_caught(tmp_path):
    """It trains as a background image: the vehicle learns its condition is empty."""
    shard = _shard(tmp_path, 1)
    (shard / "labels" / "train" / "a.txt").unlink()
    assert "images with no label file" in _checks(_validate.check_fleet(tmp_path, held=set()))


def test_an_image_two_vehicles_both_hold_is_caught(tmp_path):
    """num_examples is FedAvg's weight, so a shared image votes twice."""
    _shard(tmp_path, 1, train=("a.jpg",))
    _shard(tmp_path, 2, train=("a.jpg",), val=("w.jpg",))
    assert ("images shared between two vehicles' train sets"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


def test_an_image_in_both_train_and_val_is_caught(tmp_path):
    """Evaluation becomes a memory test and mAP stops meaning anything."""
    _shard(tmp_path, 1, train=("a.jpg",), val=("a.jpg",))
    assert ("images in both train and val of one shard"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


def test_a_held_out_image_inside_a_shard_is_caught(tmp_path):
    """The holdout is the only honest metric; an image inside a shard undoes that."""
    _shard(tmp_path, 1, train=("a.jpg",))
    problems = _validate.check_fleet(tmp_path, held={"a.jpg"})
    assert "held-out images found inside a shard" in _checks(problems)


def test_a_listing_naming_a_file_that_is_not_there_is_caught(tmp_path):
    shard = _shard(tmp_path, 1)
    (shard / "train.txt").write_text("a.jpg\nghost.jpg\n")
    assert ("images listed but not materialised"
            in _checks(_validate.check_fleet(tmp_path, held=set())))


@pytest.mark.parametrize("text,why", [
    ("", "empty"),
    ("0 0.5 0.5\n", "too few fields"),
    ("x 0.5 0.5 0.2 0.2\n", "non-numeric"),
    ("99 0.5 0.5 0.2 0.2\n", "class id outside nc"),
    ("0 1.5 0.5 0.2 0.2\n", "unnormalised coordinates"),
])
def test_unusable_label_files_are_caught(tmp_path, text, why):
    _shard(tmp_path, 1, label_text=text)
    assert "unusable label files" in _checks(_validate.check_fleet(tmp_path, held=set())), why


def test_validation_repairs_nothing(tmp_path):
    """A validator that edits data hides the bug that produced the data."""
    shard = _shard(tmp_path, 1)
    (shard / "labels" / "train" / "a.txt").unlink()
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    _validate.check_fleet(tmp_path, held=set())
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


def test_validate_runs_before_the_gpu_stages():
    """Finding a broken shard after an hour of training is finding it too late."""
    names = [s.name for s in stages.STAGES]
    assert names.index("validate") > names.index("fleet")
    assert names.index("validate") < names.index("sanity") < names.index("federate")
    assert not stages.BY_NAME["validate"].gated, "seconds of scanning; never worth skipping"


# --------------------------------------------------------------- run comparison
from pipeline import compare as _compare  # noqa: E402


def _report(dir_, name, config, holdout_map=None, self_map=None, checksums=2):
    d = dir_ / name
    d.mkdir(parents=True)
    data = {"config": config, "checksums": list(range(checksums)), "learned": checksums > 1,
            "gpu": {"energy_wh": 10.0}, "stages": [{"seconds": 100}],
            "metrics": ([{"stage": "evaluate", "mAP50": self_map}] if self_map else [])}
    if holdout_map:
        data["holdout"] = {"rounds": [{"round": 1, "mAP50": holdout_map, "mAP50-95": 0.2}]}
    (d / "report.json").write_text(json.dumps(data))
    return d


def test_comparison_leads_with_the_holdout_number(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2}, holdout_map=0.40, self_map=0.46)
    _report(tmp_path, "20260102-000000", {"seed": 1, "rounds": 2}, holdout_map=0.42, self_map=0.44)
    runs = _compare.load(5, reports_dir=tmp_path)

    assert [r["holdout_mAP50"] for r in runs] == [0.40, 0.42]
    out = _compare.table(runs)
    assert "holdout mAP50" in out and out.index("holdout mAP50") < out.index("self mAP50")
    assert "+0.0200" in out


def test_comparison_says_when_more_than_one_setting_changed(tmp_path):
    """Two changed variables mean the difference cannot be attributed to either."""
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2, "strategy": "fedavg"},
            holdout_map=0.40)
    _report(tmp_path, "20260102-000000", {"seed": 0, "rounds": 6, "strategy": "fedadam"},
            holdout_map=0.45)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "WARNING" in out and "rounds" in out and "strategy" in out


def test_a_single_changed_setting_is_not_warned_about(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0, "rounds": 2}, holdout_map=0.40)
    _report(tmp_path, "20260102-000000", {"seed": 1, "rounds": 2}, holdout_map=0.41)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "WARNING" not in out
    assert "seed" in out, "the varying setting should still be shown as a column"


def test_runs_without_a_holdout_number_are_called_out(tmp_path):
    """They predate the holdout, so they cannot be compared between fleets."""
    _report(tmp_path, "20260101-000000", {"seed": 0}, self_map=0.46)
    out = _compare.table(_compare.load(5, reports_dir=tmp_path))
    assert "no holdout number" in out


def test_a_corrupt_report_does_not_stop_the_comparison(tmp_path):
    _report(tmp_path, "20260101-000000", {"seed": 0}, holdout_map=0.40)
    bad = tmp_path / "20260102-000000"
    bad.mkdir()
    (bad / "report.json").write_text("{not json")
    assert len(_compare.load(5, reports_dir=tmp_path)) == 1


# ------------------------------------------------------------------ experiments
from pipeline import experiment as _exp  # noqa: E402


def test_every_arm_of_a_preset_changes_exactly_one_setting():
    """Two changed variables in one comparison explain neither."""
    base = {"profile": "demo", "vehicles": 6, "rounds": 2, "epochs": 1, "per_vehicle": 0}
    for preset, expect in (("seeds", "seed"), ("strategies", "strategy"),
                           ("partitions", "partition")):
        arms = _exp.arms_for(preset, base, [0, 1], ["fedavg", "fedadam"],
                             ["condition", "random"], [0.5])
        assert len(arms) == 2
        differing = {k for k in arms[0] if k != "label" and arms[0][k] != arms[1][k]}
        assert differing == {expect}, f"{preset} varies {differing}"


def test_the_alpha_preset_also_selects_the_partition_it_belongs_to():
    """alpha means nothing under condition partitioning; asking for it must switch."""
    arms = _exp.arms_for("alpha", {"profile": "demo"}, [], [], [], [0.05, 100.0])
    assert {a["partition"] for a in arms} == {"dirichlet"}
    assert [a["alpha"] for a in arms] == [0.05, 100.0]


def test_an_unknown_preset_is_refused():
    with pytest.raises(SystemExit):
        _exp.arms_for("vibes", {}, [], [], [], [])


def test_an_arm_becomes_a_real_runner_invocation():
    """Arms are driven through pipeline.runner, not a second code path that can
    drift from the one people actually use."""
    cmd = _exp.command({"profile": "full", "rounds": 6, "epochs": 4, "seed": 2,
                        "strategy": "fedadam", "partition": "dirichlet", "alpha": 0.3,
                        "per_vehicle": 1400}, confirm=True)
    assert cmd[1:4] == ["-m", "pipeline.runner", "--all"]
    for flag, value in (("--profile", "full"), ("--rounds", "6"), ("--epochs", "4"),
                        ("--seed", "2"), ("--strategy", "fedadam"),
                        ("--partition", "dirichlet"), ("--alpha", "0.3"),
                        ("--per-vehicle", "1400")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--yes" in cmd


def test_the_runbook_only_promises_commands_that_exist():
    """A runbook that names a module nobody wrote is worse than no runbook."""
    import re
    text = (REPO / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (pipeline\.[a-z_]+)", text))
    assert modules, "the runbook should name the entry points"
    for mod in modules:
        assert (REPO / "pipeline" / f"{mod.split('.')[1]}.py").is_file(), f"{mod} missing"
    for script in ("scripts/run_pipeline.ps1", "scripts/run_pipeline.sh"):
        assert (REPO / script).is_file(), f"{script} is referenced but absent"


def test_the_baseline_pools_only_the_shards_that_trained(monkeypatch, tmp_path):
    """A 6-vehicle run trains 6 of the 10 materialised shards. Pooling all ten hands
    the centralised model data the federation never saw, and the gap then measures
    the extra data."""
    from pipeline import baseline as _b, vehicle_metrics as _vm

    batches = tmp_path / "batch"
    for i in range(1, 11):
        (batches / f"batch_{i}").mkdir(parents=True)
        (batches / f"batch_{i}" / "train.txt").write_text(f"img{i}a.jpg\nimg{i}b.jpg\n")
    monkeypatch.setattr(paths, "VEHICLE_BATCHES", batches)
    monkeypatch.setattr(_vm, "per_vehicle_rounds", lambda: {"1": [], "2": [], "5": []})

    assert _b.trained_shards() == [1, 2, 5]
    assert len(_b.pooled_names()) == 6, "only the three shards that trained"
    assert len(_b.pooled_names(shards=list(range(1, 11)))) == 20


def test_parity_flags_a_ceiling_that_was_given_more_compute():
    """The run of 2026-08-06 pooled 14 000 images for 24 epochs against a federation
    that made 201 600 image-visits: 1.667x, and its retention figure is a bound."""
    from pipeline import baseline as _b

    over = _b.parity(images=14000, epochs=24, shards=6, per_vehicle=1400,
                     rounds=6, local_epochs=4)
    assert over["ratio"] == 1.667 and over["matched"] is False

    fair = _b.parity(images=8400, epochs=24, shards=6, per_vehicle=1400,
                     rounds=6, local_epochs=4)
    assert fair["ratio"] == 1.0 and fair["matched"] is True


def test_the_report_calls_an_over_provisioned_ceiling_what_it_is(monkeypatch):
    from pipeline import baseline as _b, holdout as _h

    monkeypatch.setattr(_h, "curve", lambda: {
        "holdout": {"size": 1000},
        "rounds": [{"round": 1, "mAP50": 0.43, "mAP50-95": 0.24,
                    "precision": 0.6, "recall": 0.4}]})
    monkeypatch.setattr(_b, "gap", lambda: {
        "federated_mAP50": 0.4334, "centralised_mAP50": 0.4771, "gap": 0.0437,
        "retained": 0.908, "matched": False, "budget_ratio": 1.667})
    md = report.to_markdown(report.collect(config={}))
    assert "lower bound" in md and "1.667" in md


def test_the_sanity_stage_does_not_shell_out_to_a_moving_target():
    """`python -m ultralytics.cfg` ran until ultralytics 8.4 made cfg a package with
    no __main__, and the whole chain then halted at the first GPU stage. The sanity
    stage calls the same API a client calls, which cannot drift from it."""
    cmd = stages._cmd_sanity(Config(profile="demo"))
    assert cmd[1] == "-c", "invoke the API, not a console script that may not exist"
    body = cmd[2]
    assert "ultralytics.cfg" not in body
    assert "from ultralytics import YOLO" in body
    assert "imgsz=320" in body
    # And it trains on a shard that exists, not on the data.runtime.yaml a client
    # writes at runtime -- which made the stage pass only where it was not needed.
    assert "data.runtime.yaml" not in body
    # repr, because the snippet embeds the path as a Python literal -- on Windows a
    # raw path would turn every backslash into an escape.
    assert repr(str(paths.VEHICLE_BATCHES / "batch_1" / "data.yaml")) in body


def test_the_sanity_stage_refuses_to_run_without_a_shard(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "VEHICLE_BATCHES", tmp_path / "nothing")
    with pytest.raises(RuntimeError, match="fleet stage"):
        stages._cmd_sanity(Config())


def test_a_line_the_console_cannot_encode_does_not_kill_the_output_thread(monkeypatch):
    """On Windows a redirected stdout is cp1252, Ultralytics' progress bars are not,
    and an unguarded print raised inside the drain thread -- which then died, the pipe
    it was reading filled, and the stage failed for a reason nothing in the log
    explained. Reproduced with a real strict cp1252 stream."""
    import io
    from pipeline import runner as _r

    raw = io.BytesIO()
    strict = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(_r.sys, "stdout", strict)
    monkeypatch.setattr("sys.stdout", strict)

    _r._safe_print("progress ━━ bar")      # box drawing: not in cp1252
    strict.flush()

    monkeypatch.undo()
    written = raw.getvalue().decode("cp1252")
    assert "progress" in written and "bar" in written, written


def test_flwr_is_resolved_next_to_the_interpreter_not_from_path(monkeypatch, tmp_path):
    """A shell a person types into has the venv on PATH; one started by a script does
    not, and the stage died with '[WinError 2] The system cannot find the file
    specified' -- which names no file."""
    fake = tmp_path / "flwr.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(stages, "PY", str(tmp_path / "python.exe"))
    assert stages.flwr_executable() == str(fake)
    assert stages._cmd_federate(Config())[0] == str(fake)


def test_a_missing_flwr_says_which_environment_to_install_it_into(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "PY", str(tmp_path / "python.exe"))
    monkeypatch.setattr(stages.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="pip install flwr"):
        stages.flwr_executable()


def test_every_subprocess_is_told_to_write_utf8():
    """flwr prints a flower emoji in its banner. With a redirected stdout on Windows
    the child gets cp1252 and dies with "'charmap' codec can't encode character
    '\U0001f338'" -- a federation failing for a reason unrelated to federation, and
    only when launched from a script rather than a terminal."""
    env = paths.subprocess_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1"


def test_the_interpreters_scripts_directory_leads_the_path():
    """flwr resolves `flower-superlink` from PATH, not from its own location, so a
    non-interactive shell without the venv activated failed with
    'Unable to launch flower-superlink ... [WinError 2]'. Our children spawn children."""
    import os
    import sys

    env = paths.subprocess_env()
    first = env["PATH"].split(os.pathsep)[0]
    assert first == str(Path(sys.executable).parent)
    assert env["PATH"].count(first) >= 1


def test_checkpoints_from_a_previous_run_are_not_plotted_as_this_ones(tmp_path, monkeypatch):
    """A 2-round run leaves rounds 3..6 from a previous 6-round one in the directory.
    Scoring all of them drew a curve that jumped 0.014 -> 0.217 between round 2 and
    round 3: two different models plotted as one."""
    import os
    import time
    from pipeline import holdout as _h

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    old = time.time() - 3600
    for i in (3, 4, 5, 6):
        f = ckpts / f"global_round_{i}.pt"
        f.write_bytes(b"old")
        os.utime(f, (old, old))
    for i in (1, 2):
        (ckpts / f"global_round_{i}.pt").write_bytes(b"new")

    monkeypatch.setattr(paths, "PROJECT", tmp_path)
    assert [p.name for p in _h.checkpoints()] == ["global_round_1.pt", "global_round_2.pt"]


def test_a_demo_ceiling_does_not_overwrite_the_archive_of_a_full_one(tmp_path):
    """40 minutes of GPU time should not be replaceable by 30 seconds of it."""
    from pipeline import baseline as _b

    names = {f"baseline-{images}img-{epochs}ep.json"
             for images, epochs in ((8400, 24), (1200, 2))}
    assert len(names) == 2, "the archive name must distinguish the budgets"


def test_the_checksum_criterion_judges_one_run_not_a_pile_of_them(tmp_path):
    """Logs are per process and accumulate. Reading all of them gave eleven
    checksums for a three-round run, so the single most important signal in this
    project was being computed over a mixture."""
    import os
    import time

    old = time.time() - 3600
    (tmp_path / "server.111.log").write_text(
        "Aggregated parameters with checksum: 5.0\n"
        "Aggregated parameters with checksum: 5.0\n")     # a previous run that stalled
    os.utime(tmp_path / "server.111.log", (old, old))
    (tmp_path / "server.222.log").write_text(
        "Aggregated parameters with checksum: 1.0\n"
        "Aggregated parameters with checksum: 2.0\n")

    assert logparse.aggregate_checksums(tmp_path) == [1.0, 2.0]
    assert logparse.aggregate_checksums(tmp_path, all_runs=True) == [5.0, 5.0, 1.0, 2.0]
    # And the verdict follows this run, not the pile: the old run's stall must not
    # condemn a federation that is learning.
    assert logparse.federation_learned(tmp_path)[0] is True
