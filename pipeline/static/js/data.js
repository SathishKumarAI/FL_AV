// Data view: what the fleet is actually training on, as numbers.
//
// The drawer could show one vehicle's mixture on demand; nothing showed the fleet as
// a dataset. Selecting a shard filters the class chart, the sample strip and the
// table highlight together — one selection, three views.
import { $, esc, empty, glyph, PALETTE } from "./util.js";
import { state } from "./state.js";
import { renderStrip, loadConsumed } from "./consumed.js";

const view = { data: null, shard: null, sort: { key: "vid", dir: 1 }, loading: false,
               labels: true };

export async function loadData(force) {
  if (view.loading) return;
  if (view.data && !force) return render();
  view.loading = true;
  $("dataBody").innerHTML = '<div class="skel-block" style="min-height:280px"></div>';
  try {
    view.data = await (await fetch("/api/data" + (force ? "?refresh=1" : ""))).json();
  } catch {
    $("dataBody").innerHTML = empty("The server did not answer.",
      "Is <code>python -m pipeline.server</code> still running?");
    return;
  } finally {
    view.loading = false;
  }
  render();
}

const label = (cid) => {
  const names = (view.data && view.data.class_names) || [];
  const i = Number(cid);
  return Number.isInteger(i) && names[i] ? names[i] : String(cid);
};

/** Horizontal bars, sorted, with the count and share on each row. */
function bars(counts, opts = {}) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return '<p class="hint">Nothing counted here yet.</p>';
  const total = entries.reduce((a, [, n]) => a + n, 0) || 1;
  const max = entries[0][1] || 1;
  return '<div class="mix">' + entries.slice(0, opts.limit || 20).map(([k, n]) => {
    const name = opts.classes ? label(k) : k;
    const pct = 100 * n / total;
    return `<div class="r"><div class="t"><i style="width:${(100 * n / max).toFixed(1)}%"></i>` +
      `<span>${esc(name)}</span></div>` +
      `<div class="c" title="${n} of ${total}">${pct >= 0.1 ? pct.toFixed(1) : "<0.1"}%</div></div>`;
  }).join("") + "</div>";
}

function readouts(d) {
  const fleetImages = d.fleet.reduce((a, s) => a + (s.n_train || 0), 0);
  const fleetLabels = d.fleet.reduce((a, s) => a + (s.labels || 0), 0);
  const tile = (v, unit, k, title) =>
    `<div class="readout" title="${esc(title || "")}"><span class="v">${v}</span>` +
    (unit ? `<span class="u">${unit}</span>` : "") + `<div class="k">${k}</div></div>`;
  return '<div class="readouts">' +
    tile(d.pool.val_images.toLocaleString(), "", "val images in the pool",
         d.pool.path || "the kagglehub cache") +
    tile(d.pool.indexed.toLocaleString(), "", "with BDD attributes") +
    tile(fleetImages.toLocaleString(), "", "materialised across the fleet") +
    tile(fleetLabels.toLocaleString(), "", "labelled objects") +
    tile(d.holdout.size.toLocaleString(), "", "held out, seen by nobody") +
    tile((d.pooled_baseline || 0).toLocaleString(), "", "pooled for the ceiling") +
    "</div>";
}

function shardTable(d) {
  const cols = [["vid", "vehicle"], ["condition", "condition"], ["n_train", "train"],
                ["n_val", "val"], ["labels", "objects"], ["density", "objects/img"],
                ["fingerprint", "fingerprint"]];
  const rows = d.fleet.map(s => ({...s, density: s.n_train ? s.labels / s.n_train : 0}));
  const { key, dir } = view.sort;
  rows.sort((a, b) => {
    const x = a[key], y = b[key];
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  return '<table><thead><tr>' + cols.map(([k, t]) =>
      `<th scope="col"><button class="sortbtn" data-key="${k}" ` +
      `aria-sort="${key === k ? (dir > 0 ? "ascending" : "descending") : "none"}">` +
      `${t}${key === k ? (dir > 0 ? " ▲" : " ▼") : ""}</button></th>`).join("") +
    '</tr></thead><tbody>' + rows.map(s =>
      `<tr class="shardrow" data-vid="${s.vid}" data-sel="${String(view.shard) === String(s.vid)}">` +
      `<td><button class="linky" data-vid="${s.vid}">${glyph(s.condition)} v${s.vid}</button></td>` +
      `<td>${esc(s.condition || "?")}</td><td class="num">${s.n_train ?? "—"}</td>` +
      `<td class="num">${s.n_val ?? "—"}</td><td class="num">${(s.labels || 0).toLocaleString()}</td>` +
      `<td class="num">${s.density.toFixed(1)}</td>` +
      `<td class="num" style="color:var(--dim)">${esc(s.fingerprint || "—")}</td>` +
      (s.held_out_inside ? `<td class="num warn" title="held-out images inside this shard">` +
        `${s.held_out_inside} leak</td>` : "") + "</tr>").join("") + "</tbody></table>";
}

function conditionRows(d) {
  // One row per vehicle, its weather mix as a proportional stack. The fleet's
  // non-IID-ness, visible in one glance instead of ten drawer visits.
  const keys = {};
  d.fleet.forEach(s => Object.entries(s.mix.weather || {}).forEach(([k, n]) => {
    keys[k] = (keys[k] || 0) + n;
  }));
  const order = Object.keys(keys).sort((a, b) => keys[b] - keys[a]);
  const colour = k => PALETTE[order.indexOf(k) % PALETTE.length];
  return '<div class="stacks">' + d.fleet.map(s => {
    const total = Object.values(s.mix.weather || {}).reduce((a, b) => a + b, 0) || 1;
    const segs = order.filter(k => s.mix.weather[k]).map(k =>
      `<i style="width:${(100 * s.mix.weather[k] / total).toFixed(2)}%;background:${colour(k)}" ` +
      `title="v${s.vid} ${esc(k)}: ${s.mix.weather[k]} (${(100 * s.mix.weather[k] / total).toFixed(1)}%)"></i>`
    ).join("");
    return `<div class="stackrow"><span class="lbl">v${s.vid} ${esc(s.condition || "")}</span>` +
           `<span class="stack">${segs}</span></div>`;
  }).join("") + "</div>" +
  '<div class="legend">' + order.map(k =>
    `<span><i style="background:${colour(k)}"></i>${esc(k)}</span>`).join("") + "</div>";
}

function render() {
  const d = view.data;
  if (!d) return;
  if (!d.fleet || !d.fleet.length) {
    $("dataBody").innerHTML = empty("No fleet on disk.",
      "Run the <code>fleet</code> stage, then reload this tab.");
    return;
  }
  const sel = view.shard && d.fleet.find(s => String(s.vid) === String(view.shard));
  const classes = sel ? sel.classes : d.fleet_classes;

  $("dataBody").innerHTML =
    `<div class="panel"><h2>The pool, the fleet, the holdout` +
      `<span class="n">fingerprint ${esc(d.fingerprint || "—")}</span></h2>` +
      readouts(d) +
      `<p class="hint">Images are hardlinks onto the kagglehub cache, so a fleet of ` +
      `ten shards costs no extra disk. The fingerprint is over the assignment: two ` +
      `runs carrying the same one trained on exactly the same images.</p></div>` +

    `<div class="grid two" style="grid-template-columns:1fr 1fr">` +
      `<div class="panel"><h2>Class distribution` +
        `<span class="n">${sel ? `vehicle ${sel.vid}` : "whole fleet"}</span></h2>` +
        bars(classes, {classes: true}) +
        `<p class="hint">BDD100K is dominated by <b>car</b>. A detector that predicted ` +
        `nothing else would still look respectable on mAP averaged over images, which ` +
        `is why per-class numbers matter more than the headline.` +
        (sel ? ` <button class="linky" id="clearShard">show the whole fleet</button>` : "") +
        `</p></div>` +

      `<div class="panel"><h2>Weather mix per vehicle<span class="n">non-IID, visibly</span></h2>` +
        conditionRows(d) +
        `<p class="hint">Each bar is one vehicle's shard. If these rows looked alike, ` +
        `the fleet would be learning nothing that pooling the data would not teach it.</p></div>` +
    `</div>` +

    `<div class="panel"><h2>Shards<span class="n">click a row to filter</span></h2>` +
      shardTable(d) +
      `<p class="hint">objects/img is the label density: a daytime city street holds ` +
      `far more annotated objects than a highway, which is part of why the vehicles' ` +
      `own scores are not comparable with each other.</p></div>` +

    `<div class="grid two" style="grid-template-columns:1fr 1fr">` +
      `<div class="panel"><h2>Holdout<span class="n">${d.holdout.size} images, seed ` +
        `${(d.holdout.meta || {}).seed ?? "?"}</span></h2>` +
        `<div class="readouts" style="margin-bottom:var(--s3)">` +
          `<div class="readout"><span class="v">${(d.holdout.labels || 0).toLocaleString()}</span>` +
          `<div class="k">labelled objects</div></div>` +
          `<div class="readout"><span class="v">` +
          `${(d.holdout.labels / Math.max(1, d.holdout.size)).toFixed(1)}</span>` +
          `<div class="k">objects per image</div></div></div>` +
        bars(d.holdout.classes, {classes: true, limit: 8}) +
        `<p class="hint">It has to look like the data it measures. A holdout drawn from ` +
        `one condition would answer a different question than the one being asked.</p></div>` +

      `<div class="panel"><h2>What ${sel ? "vehicle " + sel.vid : "the fleet"} sees` +
        `<button class="chip" id="toggleLabels" data-sel="${view.labels}">` +
        `${view.labels ? "labels on" : "labels off"}</button></h2>` +
        `<div class="strip" id="dataStrip"></div>` +
        `<p class="hint">Frames from the shard itself, with the label file drawn over ` +
        `them — the boxes are what the trainer reads, not a prediction. A frame whose ` +
        `boxes sit in the wrong place is a broken shard, and no histogram would say so. ` +
        `Select a vehicle in the table to change them.</p></div>` +
    `</div>` +

    `<div class="panel"><h2>What YOLO actually consumed` +
      `<span class="n">drawn by ultralytics, during the run</span></h2>` +
      `<div id="consumedBody"></div></div>`;

  wire();
  const vid = sel ? sel.vid : (d.fleet[0] || {}).vid;
  loadStrip(vid);
  loadConsumed(vid);
}

async function loadStrip(vid) {
  const host = $("dataStrip");
  if (!host || vid == null) return;
  host.innerHTML = '<div class="skel-block" style="min-height:80px"></div>';
  try {
    const v = await (await fetch(`/api/vehicle/${encodeURIComponent(vid)}`)).json();
    await renderStrip(host, vid, v.samples, view.labels);
  } catch {
    host.innerHTML = '<p class="hint">Could not load frames.</p>';
  }
}

function wire() {
  document.querySelectorAll("#dataBody .sortbtn").forEach(b => b.onclick = () => {
    const key = b.dataset.key;
    view.sort = {key, dir: view.sort.key === key ? -view.sort.dir : 1};
    render();
  });
  document.querySelectorAll("#dataBody .shardrow, #dataBody .linky[data-vid]").forEach(el => {
    el.onclick = () => {
      const vid = el.dataset.vid;
      view.shard = String(view.shard) === String(vid) ? null : vid;
      state.sel = view.shard;
      render();
    };
  });
  const clear = $("clearShard");
  if (clear) clear.onclick = () => { view.shard = null; render(); };
  const toggle = $("toggleLabels");
  // Re-renders the strip only. A full render() would refetch nothing but would throw
  // away the consumed panel's own vehicle/group selection.
  if (toggle) toggle.onclick = () => {
    view.labels = !view.labels;
    toggle.dataset.sel = String(view.labels);
    toggle.textContent = view.labels ? "labels on" : "labels off";
    loadStrip(view.shard ?? ((view.data.fleet[0] || {}).vid));
  };
  const refresh = $("dataRefresh");
  if (refresh) refresh.onclick = () => loadData(true);
}
