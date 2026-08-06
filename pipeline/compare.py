"""Compare runs: one table, the deltas, and what actually differed between them.

The next work in the plan is repeats across seeds and a strategy comparison, and
both are unreadable one JSON file at a time. This reads the reports already on disk
and lines them up.

Two rules it enforces, because a comparison is worth less than nothing when they are
broken:

* **Configs that differ in more than one setting are flagged.** A run that changed
  the strategy *and* the epoch count explains nothing about either.
* **The holdout number leads.** Comparing self-evaluated numbers between runs
  compares the conditions each fleet happened to draw as much as the models.

    python -m pipeline.compare              # the last 5 runs
    python -m pipeline.compare --last 10 --md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import paths

#: What actually defines a run for comparison purposes.
KEYS = ("profile", "n_vehicles", "rounds", "local_epochs", "seed", "partition",
        "alpha", "strategy", "per_vehicle")


def load(limit: int = 5, reports_dir: Path | None = None) -> list[dict]:
    """The most recent run reports, newest last, with their derived numbers."""
    root = reports_dir or paths.REPORTS
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir())[-limit:]:
        f = d / "report.json"
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"run": d.name, **summarise(data)})
    return out


def summarise(data: dict) -> dict:
    """One run reduced to the numbers worth putting side by side."""
    cfg = data.get("config") or {}
    holdout_rounds = ((data.get("holdout") or {}).get("rounds")) or []
    self_eval = [r.get("mAP50") for r in (data.get("metrics") or [])
                 if r.get("stage") == "evaluate" and r.get("mAP50") is not None]
    gpu = data.get("gpu") or {}
    stages = data.get("stages") or []
    return {
        "config": {k: cfg.get(k) for k in KEYS},
        "holdout_mAP50": max((r["mAP50"] for r in holdout_rounds), default=None),
        "holdout_mAP50_95": max((r["mAP50-95"] for r in holdout_rounds), default=None),
        "self_mAP50": max(self_eval, default=None),
        "rounds_done": len(data.get("checksums") or []),
        "learned": bool(data.get("learned")),
        "energy_wh": gpu.get("energy_wh"),
        "seconds": round(sum(s.get("seconds", 0) or 0 for s in stages), 1),
        "baseline_mAP50": (data.get("baseline") or {}).get("centralised_mAP50"),
    }


def differences(runs: list[dict]) -> dict[str, list]:
    """Settings that are not the same across every run being compared."""
    varying: dict[str, list] = {}
    for key in KEYS:
        values = [r["config"].get(key) for r in runs]
        if len(set(map(repr, values))) > 1:
            varying[key] = values
    return varying


def _fmt(v, nd=4):
    # ASCII on purpose: this table is printed to a Windows console, where an em dash
    # renders as a replacement character and makes a clean table look corrupt.
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def table(runs: list[dict], markdown: bool = False) -> str:
    if not runs:
        return "No run reports found. Every run writes one to pipeline/reports/."
    varying = differences(runs)
    head = ["run", "holdout mAP50", "self mAP50", "rounds", "learned", "Wh", "sec"]
    head += list(varying)
    rows = []
    for i, r in enumerate(runs):
        row = [r["run"], _fmt(r["holdout_mAP50"]), _fmt(r["self_mAP50"]),
               str(r["rounds_done"]), "yes" if r["learned"] else "NO",
               _fmt(r["energy_wh"], 1), _fmt(r["seconds"], 0)]
        row += [str(varying[k][i]) for k in varying]
        rows.append(row)

    lines = []
    if markdown:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        lines += ["| " + " | ".join(r) + " |" for r in rows]
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
        lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        lines.append("  ".join("-" * w for w in widths))
        lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]

    note = []
    with_holdout = [r for r in runs if r["holdout_mAP50"] is not None]
    if len(with_holdout) >= 2:
        first, last = with_holdout[0], with_holdout[-1]
        note.append(f"\nholdout mAP50 {first['run']} -> {last['run']}: "
                    f"{last['holdout_mAP50'] - first['holdout_mAP50']:+.4f}")
    missing = len(runs) - len(with_holdout)
    if missing:
        note.append(f"{missing} run(s) have no holdout number and are not comparable "
                    f"between fleets; run `python -m pipeline.holdout --evaluate`.")
    if len(varying) > 1:
        note.append("WARNING: these runs differ in " + ", ".join(varying) +
                    ". More than one changed setting means the difference in the "
                    "numbers cannot be attributed to any of them.")
    return "\n".join(lines + note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last", type=int, default=5, help="how many recent runs to compare")
    ap.add_argument("--md", action="store_true", help="markdown table, for pasting into docs")
    ap.add_argument("--json", action="store_true", help="the raw comparison data")
    args = ap.parse_args(argv)

    runs = load(args.last)
    if args.json:
        print(json.dumps(runs, indent=1))
        return 0
    print(table(runs, markdown=args.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
