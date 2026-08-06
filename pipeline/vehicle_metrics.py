"""How each vehicle learns from its own slice of the world.

The fleet is condition-biased on purpose: a night-driving vehicle and a highway one
see genuinely different data, so their curves should diverge. That divergence is the
whole point of federating rather than pooling — and until now it was real in the data
and invisible in the UI.

Everything here is parsed from output my-project already produces. Nothing was added
to it to make this possible.
"""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from . import logparse, paths

# "[Client] 3 Training done. metrics={'precision': 0.30, 'mAP50': 0.23, ...}"
RE_TRAIN_DONE = re.compile(r"\[Client\]\s+(\d+)\s+Training done\. metrics=(\{.*\})")

# Ultralytics prepends its settings dir to our project=, hence runs/detect/runs/fl.
RESULTS_GLOB = "runs/**/fl/batch*/results.csv"

_CSV_MAP = {
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "train/dfl_loss": "dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "val/box_loss": "val_box_loss",
    "lr/pg0": "lr",
}


def per_vehicle_rounds() -> dict[str, list[dict]]:
    """{vid: [metrics per round]}, in the order the rounds happened."""
    out: dict[str, list[dict]] = {}
    for f in logparse.iter_logs("client*.log"):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for m in RE_TRAIN_DONE.finditer(text):
            vid, blob = m.group(1), m.group(2)
            try:
                # literal_eval, never eval: this string comes from a log file, and a
                # log file is not a trust boundary worth betting the process on.
                metrics = ast.literal_eval(blob)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(metrics, dict):
                continue
            row = {"round": len(out.get(vid, [])) + 1}
            for key in ("precision", "recall", "mAP50", "mAP50-95", "fitness"):
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    row[key.replace("-", "_")] = float(val)
            out.setdefault(vid, []).append(row)
    return out


def per_vehicle_epochs() -> dict[str, list[dict]]:
    """{vid: [per-epoch rows]} from each vehicle's Ultralytics results.csv.

    Ultralytics reuses the same run directory every round, so this holds the most
    recent round's epochs -- not the whole history. Cross-round history comes from
    per_vehicle_rounds(); the UI says which is which rather than implying otherwise.
    """
    out: dict[str, list[dict]] = {}
    for csv_path in sorted(paths.PROJECT.glob(RESULTS_GLOB)):
        m = re.search(r"batch(\d+)", csv_path.parent.name)
        if not m:
            continue
        rows: list[dict] = []
        try:
            with open(csv_path, newline="") as fh:
                for raw in csv.DictReader(fh):
                    row: dict = {}
                    for src, dst in _CSV_MAP.items():
                        v = (raw.get(src) or "").strip()
                        if v:
                            try:
                                row[dst] = float(v)
                            except ValueError:
                                pass
                    ep = (raw.get("epoch") or "").strip()
                    if ep:
                        try:
                            row["epoch"] = int(float(ep))
                        except ValueError:
                            pass
                    if row:
                        rows.append(row)
        except OSError:
            continue
        if rows:
            out[m.group(1)] = rows
    return out


def weight_movement() -> dict[str, list[float]]:
    """{vid: [|sent - received| per round]} -- how far each vehicle moved the model.

    A vehicle whose movement is zero contributed nothing that round, whatever its
    metrics say.
    """
    out: dict[str, list[float]] = {}
    for f in logparse.iter_logs("client*.log"):
        try:
            events = logparse.parse_text(f.read_text(errors="replace"))
        except OSError:
            continue
        vid, received = None, None
        for ev in events:
            if ev.kind == "training_start":
                vid = str(ev.value)
            elif ev.kind == "client_received_checksum":
                received = ev.value
            elif ev.kind == "client_sent_checksum" and vid and received is not None:
                out.setdefault(vid, []).append(abs(ev.value - received))
                received = None
    return out


def divergence(rounds: dict[str, list[dict]] | None = None,
               metric: str = "mAP50") -> dict[str, list[float]]:
    """{vid: [metric - fleet mean, per round]}.

    Signed on purpose: which vehicles are ahead of the fleet and which behind is the
    interesting part, and it is exactly what non-IID data produces.
    """
    rounds = per_vehicle_rounds() if rounds is None else rounds
    if not rounds:
        return {}
    n_rounds = max(len(v) for v in rounds.values())
    out: dict[str, list[float]] = {vid: [] for vid in rounds}
    for i in range(n_rounds):
        vals = {vid: rows[i].get(metric) for vid, rows in rounds.items()
                if i < len(rows) and rows[i].get(metric) is not None}
        if not vals:
            continue
        mean = sum(vals.values()) / len(vals)
        for vid in rounds:
            if vid in vals:
                out[vid].append(round(vals[vid] - mean, 6))
    return out


def contribution(fleet: list[dict]) -> dict[str, float]:
    """{vid: share of total training images} -- literally each vehicle's FedAvg weight."""
    total = sum(v.get("n_train", 0) for v in fleet) or 1
    return {str(v["vid"]): round(v.get("n_train", 0) / total, 6) for v in fleet}


def summary() -> dict:
    """Everything the dashboard and the report need, in one call."""
    rounds = per_vehicle_rounds()
    epochs = per_vehicle_epochs()
    from . import vehicles as _vehicles
    fleet = _vehicles.load_fleet()
    conditions = {str(v["vid"]): v.get("condition", "?") for v in fleet}
    return {
        "rounds": rounds,
        "epochs": epochs,
        "movement": weight_movement(),
        "divergence": divergence(rounds),
        "contribution": contribution(fleet),
        "conditions": conditions,
        "trained": sorted(rounds, key=lambda v: int(v)),
        "epochs_note": "results.csv holds the most recent round only (Ultralytics reuses the dir)",
    }


def demo() -> None:
    s = summary()
    print("vehicles that trained:", s["trained"])
    for vid in s["trained"]:
        curve = [r.get("mAP50") for r in s["rounds"][vid]]
        div = s["divergence"].get(vid, [])
        print(f"  vehicle {vid:>2} {s['conditions'].get(vid,'?'):<22} "
              f"mAP50 {curve} divergence {div} "
              f"movement {[round(x,2) for x in s['movement'].get(vid, [])]}")


if __name__ == "__main__":
    demo()
