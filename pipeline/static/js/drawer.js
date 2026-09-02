// Per-vehicle detail drawer: curves, weight movement, shard composition, samples.
// Opened from a fleet card; Escape closes it and focus returns where it came from.
import { $, esc, fmt, glyph, PALETTE } from "./util.js";
import { lineChart } from "./chart.js";
import { state } from "./state.js";
import { renderFleet } from "./fleet.js";

let lastFocus = null;

export async function openDrawer(vid) {
  state.sel = vid;
  lastFocus = document.activeElement;
  destroy();

  const scrim = document.createElement("div");
  scrim.className = "scrim";
  scrim.onclick = () => closeDrawer();
  const aside = document.createElement("aside");
  aside.id = "drawer";
  aside.setAttribute("role", "dialog");
  aside.setAttribute("aria-modal", "true");
  aside.setAttribute("aria-label", `Vehicle ${vid} detail`);
  document.body.append(scrim, aside);
  state.drawer = { scrim, aside, vid };

  render(vid, null);                       // skeletons while the fetch is in flight
  try {
    const data = await (await fetch(`/api/vehicle/${encodeURIComponent(vid)}`)).json();
    if (state.drawer && state.drawer.vid === vid) render(vid, data);
  } catch {
    if (state.drawer) render(vid, { error: "The server did not answer. Is it still running?" });
  }
}

function destroy() {
  if (!state.drawer) return;
  state.drawer.scrim.remove();
  state.drawer.aside.remove();
  state.drawer = null;
}

export function closeDrawer() {
  const vid = state.drawer && state.drawer.vid;
  destroy();
  state.sel = null;
  renderFleet();
  // Focus goes back to the card that opened this, found by vid rather than by
  // node: renderFleet has just replaced the element the drawer was opened from,
  // so holding the old reference would drop focus to the document body.
  const card = vid != null && document.querySelector(`.chan[data-vid="${vid}"]`);
  if (card) card.focus();
  else if (lastFocus && lastFocus.isConnected) lastFocus.focus();
}

document.addEventListener("keydown", e => {
  if (e.key === "Escape" && state.drawer) closeDrawer();
});

/** One attribute breakdown: the top few values as proportional bars. */
function mixBlock(title, counts) {
  const entries = Object.entries(counts || {}).slice(0, 5);
  if (!entries.length) return "";
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return `<div style="margin-bottom:var(--s3)"><div class="cap">${title}</div><div class="mix">` +
    entries.map(([k, n]) =>
      `<div class="r"><div class="t"><i style="width:${(100 * n / total).toFixed(1)}%"></i>` +
      `<span>${esc(k)}</span></div><div class="c">${Math.round(100 * n / total)}%</div></div>`
    ).join("") + "</div></div>";
}

function render(vid, data) {
  const d = state.drawer;
  if (!d) return;
  const L = state.learning || { rounds: {}, conditions: {}, epochs: {}, divergence: {} };
  const meta = (state.fleet || []).find(v => String(v.vid) === String(vid)) || {};
  const live = state.vehicles[vid] || {};
  const rounds = (L.rounds || {})[vid] || [];
  const idx = (state.fleet || []).findIndex(v => String(v.vid) === String(vid));
  const colour = PALETTE[Math.max(0, idx) % PALETTE.length];

  d.aside.innerHTML =
    `<div class="dhead">${glyph(meta.condition)}` +
      `<div><h3>Vehicle ${esc(vid)}</h3><div class="cond">` +
      `${esc(meta.condition || L.conditions[vid] || "unknown condition")}</div></div>` +
      `<button id="drawerClose" style="margin-left:auto" aria-label="Close detail">Close</button></div>` +

    `<div class="sect"><h4>Learning</h4>` +
      `<figure><svg class="chart" id="dMap" viewBox="0 0 900 200" role="img" tabindex="0"></svg></figure>` +
      `<p class="hint" id="dMapNote"></p></div>` +

    `<div class="sect"><h4>Loss, per epoch</h4>` +
      `<figure><svg class="chart" id="dLoss" viewBox="0 0 900 200" role="img" tabindex="0"></svg></figure>` +
      `<div class="legend"><span><i style="background:#5ad1e6"></i>box</span>` +
      `<span><i style="background:#f0a92b"></i>cls</span>` +
      `<span><i style="background:#4ec9a0"></i>dfl</span></div></div>` +

    `<div class="sect"><h4>Weights this vehicle handled</h4><dl class="kvs">` +
      `<dt>received</dt><dd>${fmt(live.received)}</dd>` +
      `<dt>sent back</dt><dd>${fmt(live.sent)}</dd>` +
      `<dt>moved by</dt><dd>${live.received != null && live.sent != null
        ? fmt(live.sent - live.received) : "—"}</dd>` +
      `<dt>rounds trained</dt><dd>${live.rounds ?? 0}</dd>` +
      `<dt>device</dt><dd>${esc(live.device ?? "—")}</dd></dl></div>` +

    `<div class="sect"><h4>What its shard holds</h4>` +
      (data == null ? '<div class="skel-block" style="min-height:120px"></div>'
        : data.error ? `<p class="hint warn">${esc(data.error)}</p>`
        : `<dl class="kvs" style="margin-bottom:var(--s3)"><dt>train images</dt>` +
          `<dd>${data.n_train}</dd><dt>val images</dt><dd>${data.n_val}</dd></dl>` +
          (data.indexed
            ? mixBlock("weather", data.counts.weather) +
              mixBlock("scene", data.counts.scene) +
              mixBlock("time of day", data.counts.timeofday)
            : '<p class="hint">No attribute index is cached, so the mixture cannot be counted. ' +
              'Build one with <code>python -m pipeline.build_fleet</code>.</p>')) +
      `</div>` +

    `<div class="sect"><h4>What this vehicle sees</h4>` +
      (data == null ? '<div class="skel-block" style="min-height:90px"></div>'
        : (data.samples && data.samples.length)
          ? '<div class="strip">' + data.samples.slice(0, 8).map(n =>
              `<img loading="lazy" src="/api/shard-image/${encodeURIComponent(vid)}/` +
              `${encodeURIComponent(n)}" alt="A ${esc(meta.condition || "driving")} image from ` +
              `vehicle ${esc(vid)}'s shard">`).join("") + "</div>"
          : '<p class="hint">No images are materialised in this shard yet.</p>') +
      `</div>`;

  const mapValues = rounds.map(r => r.mAP50);
  lineChart("dMap", {
    series: [{ label: "mAP50", color: colour, values: mapValues, area: true },
             { label: "mAP50-95", color: "var(--dim)", values: rounds.map(r => r.mAP50_95), dashed: true }],
    aria: `Vehicle ${vid} mAP by round`, yFmt: v => v.toFixed(3),
  });
  $("dMapNote").textContent = mapValues.filter(v => v != null).length
    ? "By round: " + mapValues.filter(v => v != null).map(v => v.toFixed(4)).join(" → ")
    : "This vehicle has not reported an evaluation yet.";

  const ep = (L.epochs || {})[vid] || [];
  lineChart("dLoss", {
    series: [{ label: "box", color: "#5ad1e6", values: ep.map(r => r.box_loss) },
             { label: "cls", color: "#f0a92b", values: ep.map(r => r.cls_loss) },
             { label: "dfl", color: "#4ec9a0", values: ep.map(r => r.dfl_loss) }],
    xLabel: i => "e" + (i + 1), aria: `Vehicle ${vid} loss per epoch`,
  });

  const close = $("drawerClose");
  if (close) { close.onclick = () => closeDrawer(); close.focus(); }
  renderFleet();
}
