// The fleet grid and the two charts that compare vehicles to each other.
import { $, esc, empty, glyph, PALETTE } from "./util.js";
import { lineChart, sparkline, ring } from "./chart.js";
import { state } from "./state.js";
import { openDrawer } from "./drawer.js";

/** vid -> [mAP50 per round], from the server's per-vehicle learning summary. */
function seriesByVehicle() {
  const L = state.learning, out = {};
  if (L && L.trained) {
    for (const vid of L.trained) {
      out[vid] = (L.rounds[vid] || []).map(r => r.mAP50).filter(v => v != null);
    }
  }
  return out;
}

/** Latest mAP50-95 per vehicle. Reported beside mAP50 everywhere, because mAP50
 *  alone hides localisation quality and this project parsed it without showing it. */
function strictByVehicle() {
  const L = state.learning, out = {};
  if (L && L.trained) {
    for (const vid of L.trained) {
      const vals = (L.rounds[vid] || []).map(r => r.mAP50_95).filter(v => v != null);
      out[vid] = vals.length ? vals[vals.length - 1] : null;
    }
  }
  return out;
}

export function renderFleet() {
  const fleet = state.fleet, L = state.learning, host = $("fleet");
  if (!host) return;
  if (!fleet || !fleet.length) {
    host.innerHTML = empty("No fleet built yet.",
      "Run the <code>fleet</code> stage — it gives each vehicle a condition-biased shard.");
    $("fleetCount").textContent = "";
    return;
  }

  const series = seriesByVehicle();
  const strict = strictByVehicle();
  const all = Object.values(series).flat();
  const lo = all.length ? Math.min.apply(null, all) : 0;
  const hi = all.length ? Math.max.apply(null, all) : 1;
  $("fleetCount").textContent =
    `${fleet.length} vehicles · ${L && L.trained ? L.trained.length : 0} trained`;

  host.innerHTML = fleet.map((v, i) => {
    const vid = String(v.vid);
    const live = state.vehicles[vid] || {};
    const vals = series[vid] || [];
    const colour = PALETTE[i % PALETTE.length];
    const delta = vals.length > 1 ? vals[vals.length - 1] - vals[0] : null;
    const share = ((L && L.contribution || {})[vid] || 0) * 100;
    const rounds = state.cfg.rounds || Math.max(1, vals.length);
    const chip = delta == null
      ? '<span class="chip flat">no Δ yet</span>'
      : `<span class="chip ${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "▲" : "▼"}` +
        `${Math.abs(delta).toFixed(4)}</span>`;
    return `<button class="chan" data-vid="${esc(vid)}" data-active="${!!live.training}" ` +
      `aria-haspopup="dialog" aria-label="Vehicle ${esc(vid)}, ${esc(v.condition)}. Open detail.">` +
      `<div class="top">${glyph(v.condition)}` +
        `<div><div class="who">Vehicle ${esc(vid)}</div>` +
        `<div class="cond">${esc(v.condition)}</div></div>` +
        `${ring(vals.length / rounds, live.training)}</div>` +
      (vals.length ? sparkline(vals, lo, hi, colour)
        : '<div class="spark" style="height:40px;display:flex;align-items:center;' +
          'font-size:var(--t-xs);color:var(--dim)">not trained yet</div>') +
      `<div class="foot"><span>mAP <b>${vals.length ? vals[vals.length - 1].toFixed(4) : "—"}</b>` +
      `<span title="mAP50-95"> / ${strict[vid] != null ? strict[vid].toFixed(3) : "—"}</span></span>` +
      `${chip}<span>${v.n_train ?? "—"} img · ${share.toFixed(0)}%</span></div></button>`;
  }).join("");

  host.querySelectorAll(".chan").forEach(el => { el.onclick = () => openDrawer(el.dataset.vid); });
  drawFleetCharts(series);
}

function drawFleetCharts(series) {
  const L = state.learning;
  if (!L || !L.trained || !L.trained.length) {
    lineChart("fleetChart", { series: [] });
    lineChart("divChart", { series: [] });
    $("fleetLegend").innerHTML = "<span>No vehicle has trained yet.</span>";
    $("divNote").textContent = "Divergence appears once a round has been aggregated.";
    return;
  }

  lineChart("fleetChart", {
    focus: state.sel,
    series: L.trained.filter(vid => !state.hidden.has(vid)).map(vid => ({
      key: vid,
      label: "v" + vid,
      color: PALETTE[L.trained.indexOf(vid) % PALETTE.length],
      values: series[vid] || [],
    })),
    aria: "mAP50 by round for each vehicle",
    yFmt: v => v.toFixed(3),
  });

  $("fleetLegend").innerHTML = L.trained.map((vid, i) =>
    `<button type="button" aria-pressed="${!state.hidden.has(vid)}" data-vid="${esc(vid)}">` +
    `<i style="background:${PALETTE[i % PALETTE.length]}"></i>v${esc(vid)} · ` +
    `${esc(L.conditions[vid] || "?")}</button>`).join("");
  $("fleetLegend").querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      const vid = b.dataset.vid;
      state.hidden.has(vid) ? state.hidden.delete(vid) : state.hidden.add(vid);
      renderFleet();
    };
  });

  drawDivergence(L.trained, L.trained.map(v => (L.divergence[v] || []).slice(-1)[0] || 0), L);
}

/** Signed distance from the fleet mean on the latest round, as bars from a zero line. */
function drawDivergence(vids, last, L) {
  const svg = $("divChart"), W = 900, H = 190, P = { l: 52, r: 14, t: 14, b: 30 };
  const mx = Math.max.apply(null, [0.0001].concat(last.map(Math.abs)));
  const mid = P.t + (H - P.t - P.b) / 2;
  const bw = (W - P.l - P.r) / Math.max(1, vids.length) * 0.5;

  let g = "";
  for (const v of [mx, 0, -mx]) {
    const yy = mid - (v / mx) * (mid - P.t);
    g += `<line x1="${P.l}" y1="${yy}" x2="${W - P.r}" y2="${yy}" ` +
         `stroke="${v === 0 ? "var(--line)" : "var(--line-soft)"}"/>` +
         `<text x="${P.l - 8}" y="${yy + 3.5}" text-anchor="end" fill="var(--dim)" ` +
         `font-size="10" font-family="var(--mono)">${v >= 0 ? "+" : ""}${v.toFixed(3)}</text>`;
  }
  vids.forEach((vid, i) => {
    const d = last[i];
    const cx = P.l + (i + 0.5) * (W - P.l - P.r) / Math.max(1, vids.length);
    const h = Math.abs(d) / mx * (mid - P.t);
    g += `<rect x="${cx - bw / 2}" y="${d >= 0 ? mid - h : mid}" width="${bw}" ` +
         `height="${Math.max(1.5, h)}" rx="2" fill="${d >= 0 ? "var(--ok)" : "var(--bad)"}" ` +
         `opacity="${state.sel && state.sel !== vid ? 0.3 : 0.9}"/>` +
         `<text x="${cx}" y="${H - 10}" fill="var(--dim)" font-size="10" ` +
         `font-family="var(--mono)" text-anchor="middle">v${esc(vid)}</text>`;
  });
  svg.innerHTML = g;
  svg._spec = null;   // bars, not a line series: no cursor readout

  const best = vids[last.indexOf(Math.max.apply(null, last))];
  const worst = vids[last.indexOf(Math.min.apply(null, last))];
  svg.setAttribute("aria-label",
    `Divergence from the fleet mean. Ahead: vehicle ${best}. Behind: vehicle ${worst}.`);
  $("divNote").innerHTML =
    `Ahead of the fleet: <b>${esc(L.conditions[best] || best)}</b>. ` +
    `Behind it: <b>${esc(L.conditions[worst] || worst)}</b>. Signed difference from the fleet ` +
    `mean on the latest round — spread is what makes this federated rather than distributed.`;
}
