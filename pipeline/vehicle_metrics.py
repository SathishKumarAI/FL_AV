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
    for f in logparse.current_run_logs("client*.log"):
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


def _fleet_built_at() -> float:
    """When the current fleet was defined. Anything older belongs to a previous run."""
    f = paths.VEHICLE_ROOT / "fleet.json"
    try:
        return f.stat().st_mtime
    except OSError:
        return 0.0


def per_vehicle_epochs(since: float | None = None) -> dict[str, list[dict]]:
    """{vid: [per-epoch rows]} from each vehicle's Ultralytics results.csv.

    Ultralytics reuses the same run directory every round, so this holds the most
    recent round's epochs -- not the whole history. Cross-round history comes from
    per_vehicle_rounds(); the UI says which is which rather than implying otherwise.

    Files older than the current fleet are skipped. Without that, a results.csv left
    behind by an earlier run gets folded into the analysis: a 20-epoch curve from a
    10-image fixture was being reported as a vehicle that "overfit", in a run where
    that vehicle trained for one epoch on 300 images.
    """
    cutoff = _fleet_built_at() if since is None else since
    out: dict[str, list[dict]] = {}
    for csv_path in sorted(paths.PROJECT.glob(RESULTS_GLOB)):
        m = re.search(r"batch(\d+)", csv_path.parent.name)
        if not m:
            continue
        try:
            if cutoff and csv_path.stat().st_mtime < cutoff:
                continue
        except OSError:
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
    for f in logparse.current_run_logs("client*.log"):
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


# --------------------------------------------------------------------------
# Fit diagnosis
# --------------------------------------------------------------------------
def fit_diagnosis(epochs: dict[str, list[dict]] | None = None,
                  rounds: dict[str, list[dict]] | None = None) -> dict:
    """Underfitting or overfitting? Say which, from train vs val loss.

    The distinction decides what to change, and guessing wrong wastes GPU hours:

    * val loss still falling with train loss, both high  -> UNDERFIT: train longer.
    * val loss rising while train loss falls             -> OVERFIT: stop earlier,
      augment more, or shrink the model.

    A low mAP on its own says neither. It is the *gap* and the *direction* that do.
    """
    epochs = per_vehicle_epochs() if epochs is None else epochs
    rounds = per_vehicle_rounds() if rounds is None else rounds

    per: dict[str, dict] = {}
    for vid, rows in epochs.items():
        tr = [r.get("box_loss") for r in rows if r.get("box_loss") is not None]
        va = [r.get("val_box_loss") for r in rows if r.get("val_box_loss") is not None]
        if not tr or not va:
            continue
        gap = va[-1] - tr[-1]
        # Direction over the last half of the epochs, which is where divergence shows.
        half = max(1, len(va) // 2)
        val_trend = va[-1] - va[-half]
        train_trend = tr[-1] - tr[-half]
        if val_trend > 0 and train_trend < 0:
            verdict = "overfitting"
        elif val_trend < 0 and train_trend < 0:
            verdict = "still learning"
        else:
            verdict = "flat"
        per[vid] = {"train_loss": round(tr[-1], 4), "val_loss": round(va[-1], 4),
                    "gap": round(gap, 4), "val_trend": round(val_trend, 4),
                    "train_trend": round(train_trend, 4), "verdict": verdict,
                    "epochs": len(rows)}

    # Fleet-level: is the aggregate still improving round over round?
    finals = [rows[-1].get("mAP50") for rows in rounds.values()
              if rows and rows[-1].get("mAP50") is not None]
    firsts = [rows[0].get("mAP50") for rows in rounds.values()
              if rows and rows[0].get("mAP50") is not None]
    improving = bool(finals and firsts and sum(finals)/len(finals) > sum(firsts)/len(firsts))
    n_rounds = max((len(r) for r in rounds.values()), default=0)

    if improving and n_rounds < 10:
        fleet = ("UNDERFIT — mAP is still climbing every round and the run stopped early. "
                 "More rounds x local epochs is the fix, not a different model.")
    elif not improving and any(p["verdict"] == "overfitting" for p in per.values()):
        fleet = "OVERFIT — validation loss is rising while training loss falls."
    elif not improving:
        fleet = "PLATEAU — the aggregate stopped improving; change LR or data, not duration."
    else:
        fleet = "IMPROVING — keep going."
    return {"per_vehicle": per, "fleet": fleet, "rounds": n_rounds, "improving": improving}
