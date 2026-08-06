"""End-of-run report: inputs, what each vehicle learned, what it cost.

Emitted as HTML (self-contained, inline SVG, no CDN — it survives being emailed) and
as Markdown (diffable, reviewable in git). Both are rendered from the same dict, so
they cannot drift.

    python -m pipeline.report --open
"""
from __future__ import annotations

import argparse
import html
import json
import platform
import time
from pathlib import Path

from . import baseline, gpu, holdout, logparse, paths, vehicle_metrics, vehicles


def collect(config: dict | None = None, telemetry: dict | None = None,
            results: list[dict] | None = None) -> dict:
    """Gather everything the report shows, from files on disk."""
    from .verify import _metrics_csv
    checksums = logparse.aggregate_checksums()
    rows = logparse.read_metrics_csv(_metrics_csv())
    learned, detail = logparse.federation_learned()

    per_vehicle: dict[int, dict] = {}
    for f in logparse.iter_logs("client*.log"):
        current = None
        for ev in logparse.parse_text(f.read_text(errors="replace")):
            if ev.kind == "training_start":
                current = int(ev.value)
                per_vehicle.setdefault(current, {"rounds": 0, "received": [], "sent": []})
                per_vehicle[current]["rounds"] += 1
            elif current and ev.kind == "client_received_checksum":
                per_vehicle[current]["received"].append(ev.value)
            elif current and ev.kind == "client_sent_checksum":
                per_vehicle[current]["sent"].append(ev.value)

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "config": config or {},
        "fleet": vehicles.load_fleet(),
        "per_vehicle": per_vehicle,
        "checksums": checksums,
        "learned": learned,
        "learned_detail": detail,
        "metrics": rows,
        "gpu": telemetry or {},
        "stages": results or [],
        "checkpoints": sorted(p.name for p in (paths.PROJECT / "checkpoints").glob("global_*.pt")),
        "learning": vehicle_metrics.summary(),
        # The only numbers not measured by a client on its own distribution.
        "holdout": holdout.curve(),
        "baseline": baseline.gap(),
    }


# --------------------------------------------------------------------------
def _svg_line(values: list[float], width: int = 720, height: int = 180,
              label: str = "") -> str:
    if not values:
        return "<p><em>no data</em></p>"
    pad, n = 28, len(values)
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    x = lambda i: pad + (0 if n == 1 else i * (width - 2 * pad) / (n - 1))
    y = lambda v: height - pad - (v - lo) / span * (height - 2 * pad)
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="#4493f8"/>'
                   for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">'
            f'<polyline points="{pts}" fill="none" stroke="#4493f8" stroke-width="2"/>{dots}</svg>')


def _num(v, suffix: str = "") -> str:
    """Missing telemetry should read as '—', never as the literal None."""
    return "—" if v is None else f"{v}{suffix}"


def to_markdown(d: dict) -> str:
    L: list[str] = [f"# Federated YOLOv8 — run report", "",
                    f"Generated {d['generated']} · {d['host']['platform']} · "
                    f"Python {d['host']['python']}", ""]

    L += ["## Inputs", "", "| setting | value |", "|---|---|"]
    for k, v in (d.get("config") or {}).items():
        L.append(f"| {k} | {v} |")
    L.append("")

    if d["fleet"]:
        L += ["## Fleet", "", "| vehicle | condition | train | val |", "|---|---|---|---|"]
        for v in d["fleet"]:
            L.append(f"| {v['vid']} | {v['condition']} | {v.get('n_train','?')} | {v.get('n_val','?')} |")
        L.append("")

    verdict = "**learned**" if d["learned"] else "**DID NOT LEARN**"
    L += ["## Did the federation learn?", "", f"{verdict} — {d['learned_detail']}", ""]
    if d["checksums"]:
        L += ["| round | aggregate checksum |", "|---|---|"]
        L += [f"| {i} | {c} |" for i, c in enumerate(d["checksums"], 1)] + [""]

    hold = (d.get("holdout") or {}).get("rounds") or []
    if hold:
        size = ((d.get("holdout") or {}).get("holdout") or {}).get("size", "?")
        L += ["## The honest global metric", "",
              f"Scored on a shared holdout of {size} images that no vehicle trained or "
              f"self-evaluated on. Every other number in this report was measured by a "
              f"client on its own distribution, so this is the one that can be compared "
              f"between runs.", "",
              "| round | mAP50 | mAP50-95 | precision | recall |", "|---|---|---|---|---|"]
        for r in hold:
            L.append(f"| {r.get('round')} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} |")
        L.append("")
        gap = d.get("baseline") or {}
        if gap:
            L += [f"Centralised ceiling on the same images: **{gap['centralised_mAP50']:.4f}**. "
                  f"The federation reaches **{gap['federated_mAP50']:.4f}**, retaining "
                  f"**{100 * gap['retained']:.1f}%** of it (gap {gap['gap']:+.4f}).", ""]
        else:
            L += ["No centralised baseline for this budget yet, so this number still has "
                  "no scale — run `python -m pipeline.baseline`.", ""]

    if d["metrics"]:
        keys = ["round", "stage", "loss", "precision", "recall", "mAP50", "mAP50-95"]
        L += ["## Metrics — per client, on its own split", "", "| " + " | ".join(keys) + " |",
              "|" + "---|" * len(keys)]
        for r in d["metrics"]:
            L.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        L.append("")

    lrn = d.get("learning") or {}
    if lrn.get("trained"):
        L += ["## How each vehicle learned", "",
              "Each vehicle trains a condition-biased, disjoint slice, so divergence "
              "between them is the expected result rather than a defect — it is what "
              "makes this federated rather than merely distributed.", "",
              "| vehicle | condition | mAP50 by round | final | final mAP50-95 | "
              "vs fleet mean | weight moved | share |",
              "|---|---|---|---|---|---|---|---|"]
        for vid in lrn["trained"]:
            rounds = lrn["rounds"].get(vid, [])
            curve = [r.get("mAP50") for r in rounds if r.get("mAP50") is not None]
            # mAP50-95 is the metric a detection paper reports; mAP50 alone hides
            # localisation quality, and this project had it parsed but never shown.
            strict = [r.get("mAP50_95") for r in rounds if r.get("mAP50_95") is not None]
            div = lrn["divergence"].get(vid, [])
            mov = lrn["movement"].get(vid, [])
            L.append(
                f"| {vid} | {lrn['conditions'].get(vid, '?')} | "
                f"{' → '.join(f'{c:.4f}' for c in curve) or '—'} | "
                f"{curve[-1]:.4f} | {f'{strict[-1]:.4f}' if strict else '—'} | "
                f"{div[-1]:+.4f} | "
                f"{(sum(mov)/len(mov)):.1f} | {lrn['contribution'].get(vid, 0)*100:.1f}% |"
                if curve else
                f"| {vid} | {lrn['conditions'].get(vid, '?')} | — | — | — | — | — | — |")
        L.append("")
        best = max(lrn["divergence"].items(), key=lambda kv: kv[1][-1] if kv[1] else -9)
        worst = min(lrn["divergence"].items(), key=lambda kv: kv[1][-1] if kv[1] else 9)
        L += [f"Strongest: **vehicle {best[0]} ({lrn['conditions'].get(best[0],'?')})** at "
              f"{best[1][-1]:+.4f} vs the fleet mean. Weakest: **vehicle {worst[0]} "
              f"({lrn['conditions'].get(worst[0],'?')})** at {worst[1][-1]:+.4f}.", ""]

    g = d.get("gpu") or {}
    if g:
        L += ["## Cost", "", "| metric | value |", "|---|---|",
              f"| GPU energy | {_num(g.get('energy_wh'), ' Wh')} |",
              f"| peak power | {_num(g.get('peak_power_w'), ' W')} |",
              f"| peak VRAM | {_num(g.get('peak_mem_mib'), ' MiB')} of {gpu.VRAM_CEILING_MIB} "
              f"({_num(g.get('peak_mem_pct'), '%')}) |",
              f"| mean utilisation | {_num(g.get('mean_util_pct'), ' %')} |", ""]

    if d["stages"]:
        L += ["## Stages", "", "| stage | status | seconds |", "|---|---|---|"]
        L += [f"| {s.get('name')} | {s.get('status')} | {round(s.get('seconds', 0), 1)} |"
              for s in d["stages"]] + [""]

    L += ["## Outputs", ""]
    L += [f"- `checkpoints/{c}`" for c in d["checkpoints"]] or ["- none written"]
    return "\n".join(L) + "\n"


def to_html(d: dict) -> str:
    maps = [r["mAP50"] for r in d["metrics"] if r.get("stage") == "evaluate" and r.get("mAP50") is not None]
    g = d.get("gpu") or {}
    fleet_rows = "".join(
        f"<tr><td>{v['vid']}</td><td>{html.escape(v['condition'])}</td>"
        f"<td>{v.get('n_train','?')}</td><td>{v.get('n_val','?')}</td></tr>" for v in d["fleet"])
    metric_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(k,'')))}</td>"
                         for k in ("round","stage","loss","precision","recall","mAP50","mAP50-95"))
        + "</tr>" for r in d["metrics"])
    stage_rows = "".join(
        f"<tr><td>{html.escape(str(s.get('name')))}</td><td>{html.escape(str(s.get('status')))}</td>"
        f"<td>{round(s.get('seconds',0),1)}</td></tr>" for s in d["stages"])
    cfg_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                       for k, v in (d.get("config") or {}).items())
    lrn = d.get("learning") or {}
    learn_rows = ""
    for vid in lrn.get("trained", []):
        curve = [r.get("mAP50") for r in lrn["rounds"].get(vid, []) if r.get("mAP50") is not None]
        div = lrn["divergence"].get(vid, [])
        mov = lrn["movement"].get(vid, [])
        learn_rows += (
            f"<tr><td>{vid}</td><td>{html.escape(str(lrn['conditions'].get(vid, '?')))}</td>"
            f"<td>{' → '.join(f'{c:.4f}' for c in curve) or '—'}</td>"
            f"<td>{div[-1]:+.4f}</td>" if div else f"<tr><td>{vid}</td><td>?</td><td>—</td><td>—</td>")
        learn_rows += (f"<td>{(sum(mov)/len(mov)):.1f}</td>" if mov else "<td>—</td>")
        learn_rows += f"<td>{lrn.get('contribution', {}).get(vid, 0)*100:.1f}%</td></tr>"

    verdict_cls = "ok" if d["learned"] else "bad"
    verdict = "The federation learned" if d["learned"] else "The federation did NOT learn"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Federated YOLOv8 — run report</title><style>
:root{{--bg:#0d1117;--panel:#151b23;--line:#242c37;--ink:#e6edf3;--dim:#8b949e;
--accent:#4493f8;--ok:#3fb950;--bad:#f85149}}
@media (prefers-color-scheme:light){{:root{{--bg:#f6f8fa;--panel:#fff;--line:#d1d9e0;--ink:#1f2328;--dim:#59636e}}}}
body{{margin:0;padding:32px;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
main{{max-width:900px;margin:0 auto}} h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:28px 0 10px}}
.sub{{color:var(--dim);font-size:13px;margin-bottom:24px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase}}
.verdict{{font-size:16px;font-weight:600;padding:12px 16px;border-radius:8px;border:1px solid}}
.verdict.ok{{color:var(--ok);border-color:var(--ok)}} .verdict.bad{{color:var(--bad);border-color:var(--bad)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}}
.metric{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.metric .n{{font-size:19px;font-weight:600;font-family:ui-monospace,monospace}}
.metric .l{{font-size:11px;color:var(--dim)}} svg{{width:100%;height:auto}}
code{{font-family:ui-monospace,monospace;font-size:12px}}
</style></head><body><main>
<h1>Federated YOLOv8 — run report</h1>
<p class="sub">{d['generated']} · {html.escape(d['host']['platform'])} · Python {d['host']['python']}</p>

<div class="panel"><div class="verdict {verdict_cls}">{verdict}</div>
<p style="color:var(--dim);font-size:13px;margin-bottom:0">{html.escape(d['learned_detail'])}</p></div>

<h2>Inputs</h2><div class="panel"><table>{cfg_rows or '<tr><td>—</td><td></td></tr>'}</table></div>

<h2>Fleet</h2><div class="panel"><table>
<thead><tr><th>vehicle</th><th>condition</th><th>train</th><th>val</th></tr></thead>
<tbody>{fleet_rows or '<tr><td colspan=4>no fleet</td></tr>'}</tbody></table></div>

<h2>Aggregate weight checksum per round</h2><div class="panel">
{_svg_line(d['checksums'], label='aggregate checksum per round')}</div>

<h2>Evaluated mAP50 per round</h2><div class="panel">
{_svg_line(maps, label='mAP50 per round')}</div>

<h2>How each vehicle learned</h2><div class="panel">
<p style="color:var(--dim);font-size:13px;margin-top:0">Each vehicle trains a
condition-biased, disjoint slice, so divergence between them is the expected result
rather than a defect.</p>
<table><thead><tr><th>vehicle</th><th>condition</th><th>mAP50 by round</th>
<th>vs fleet mean</th><th>weight moved</th><th>share</th></tr></thead>
<tbody>{learn_rows}</tbody></table></div>

<h2>Metrics</h2><div class="panel"><table>
<thead><tr><th>round</th><th>stage</th><th>loss</th><th>precision</th><th>recall</th>
<th>mAP50</th><th>mAP50-95</th></tr></thead>
<tbody>{metric_rows or '<tr><td colspan=7>none</td></tr>'}</tbody></table></div>

<h2>Cost</h2><div class="panel"><div class="metrics">
<div class="metric"><div class="n">{g.get('energy_wh', 0)}</div><div class="l">GPU energy Wh</div></div>
<div class="metric"><div class="n">{g.get('peak_power_w', 0)}</div><div class="l">peak power W</div></div>
<div class="metric"><div class="n">{g.get('peak_mem_mib', 0)}</div><div class="l">peak VRAM MiB</div></div>
<div class="metric"><div class="n">{g.get('mean_util_pct', 0)}</div><div class="l">mean util %</div></div>
</div></div>

<h2>Stages</h2><div class="panel"><table>
<thead><tr><th>stage</th><th>status</th><th>seconds</th></tr></thead>
<tbody>{stage_rows or '<tr><td colspan=3>none</td></tr>'}</tbody></table></div>

<h2>Outputs</h2><div class="panel">
{''.join(f'<div><code>checkpoints/{html.escape(c)}</code></div>' for c in d['checkpoints'])
 or '<div>no checkpoints written</div>'}</div>
</main></body></html>"""


def write(d: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (paths.REPORTS / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path, md_path = out_dir / "report.html", out_dir / "report.md"
    html_path.write_text(to_html(d), encoding="utf-8")
    md_path.write_text(to_markdown(d), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    return html_path, md_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true", help="open the HTML report afterwards")
    args = ap.parse_args(argv)
    h, m = write(collect())
    print(f"wrote {h}\nwrote {m}")
    if args.open:
        import webbrowser
        webbrowser.open(h.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
