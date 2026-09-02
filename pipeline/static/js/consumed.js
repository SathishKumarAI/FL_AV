// What YOLO consumed, as pictures. Owns two things and nothing else:
//
//   1. the label overlay on a shard frame  — a unit-square SVG over the same <img>,
//      so there is no second copy of the frame on the wire and no server-side
//      drawing library anywhere in this project;
//   2. the gallery of pictures ultralytics already drew during the last round.
//
// It draws nothing itself. Every image here came from the trainer or from the shard.
import { $, esc, empty, PALETTE } from "./util.js";

const seen = { names: [], boxes: new Map() };

/** Cache is per frame, not per vehicle: switching vehicles back and forth is common. */
async function boxesFor(vid, name) {
  const key = `${vid}/${name}`;
  if (seen.boxes.has(key)) return seen.boxes.get(key);
  let d = { boxes: [] };
  try {
    d = await (await fetch(`/api/shard-labels/${encodeURIComponent(vid)}/${encodeURIComponent(name)}`)).json();
  } catch { /* an unreachable server is already reported by the panel around this */ }
  if (d.class_names && d.class_names.length) seen.names = d.class_names;
  seen.boxes.set(key, d.boxes || []);
  return d.boxes || [];
}

const cname = (c) => seen.names[c] || `class ${c}`;

/** One frame with its labels drawn on it, or plain if `show` is false. */
function figure(vid, name, boxes, show) {
  const img = `<img loading="lazy" src="/api/shard-image/${encodeURIComponent(vid)}/${encodeURIComponent(name)}" ` +
              `alt="A frame from vehicle ${esc(String(vid))}'s shard">`;
  if (!show) return `<figure class="ovfig">${img}</figure>`;
  // viewBox is the unit square and preserveAspectRatio is off, so the SVG stretches
  // to whatever box the CSS gives the image. YOLO coordinates are already normalised,
  // which is the whole reason this needs no pixel dimensions from anywhere.
  const rects = boxes.map(b => {
    const c = PALETTE[b.cls % PALETTE.length];
    return `<rect x="${(b.cx - b.w / 2).toFixed(5)}" y="${(b.cy - b.h / 2).toFixed(5)}" ` +
           `width="${b.w.toFixed(5)}" height="${b.h.toFixed(5)}" fill="none" stroke="${c}" ` +
           `stroke-width="0.004" vector-effect="non-scaling-stroke"><title>${esc(cname(b.cls))}</title></rect>`;
  }).join("");
  return `<figure class="ovfig">${img}` +
         `<svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">${rects}</svg>` +
         `<figcaption>${boxes.length} object${boxes.length === 1 ? "" : "s"}</figcaption></figure>`;
}

/**
 * Render a strip of shard frames into `host`, optionally with the labels drawn.
 * `names` comes from /api/vehicle/<vid>, which the caller already has.
 */
export async function renderStrip(host, vid, names, show) {
  if (!host || vid == null) return;
  const frames = (names || []).slice(0, 8);
  if (!frames.length) {
    host.innerHTML = '<p class="hint">No images materialised in this shard.</p>';
    return;
  }
  const boxes = show ? await Promise.all(frames.map(n => boxesFor(vid, n))) : [];
  host.innerHTML = frames.map((n, i) => figure(vid, n, boxes[i] || [], show)).join("");
}

// -- the live feed ------------------------------------------------------------
//
// Ultralytics rewrites train_batch{0,1,2}.jpg at the start of every `train()`, and a
// round is one `train()` per vehicle. So the file on disk IS the batch the vehicle
// currently on the GPU is working through — no instrumentation, no hook, nothing
// added to the ⚠ client. The mtime is the cache-buster: same URL, new picture.

const feed = { at: 0, data: null, vid: null };
const FEED_MS = 5000;   // the page polls every 2s; re-globbing ten run dirs that often
                        // is filesystem work for a picture that changes once a round.

export async function renderNowTraining(host, who, whoLabel, busy) {
  if (!host) return;
  if (!busy || who == null) {
    host.innerHTML = '<p class="hint">Nothing is training. During a run this shows the ' +
      'batch the vehicle on the GPU is working through — the same file ultralytics ' +
      'writes for itself, not a re-render.</p>';
    feed.at = 0;
    return;
  }
  const now = performance.now();
  if (now - feed.at > FEED_MS || feed.vid !== String(who)) {
    feed.at = now;
    feed.vid = String(who);
    try {
      feed.data = await (await fetch("/api/train-artifacts")).json();
    } catch { return; }
  }
  const v = ((feed.data || {}).vehicles || []).find(x => String(x.vid) === String(who));
  const shots = (v ? v.files : []).filter(f => f.group === "consumed" && f.name.startsWith("train_batch"));
  if (!shots.length) {
    host.innerHTML = `<p class="hint">${esc(whoLabel)} has started, but has not written its ` +
      `first batch mosaic yet. It appears within the first epoch.</p>`;
    return;
  }
  host.innerHTML = '<div class="gallery">' + shots.map(f =>
    `<figure class="shot"><img src="/api/train-artifact/${encodeURIComponent(who)}/` +
    `${encodeURIComponent(f.name)}?t=${f.mtime}" alt="${esc(f.caption)}">` +
    `<figcaption>${esc(f.name)}</figcaption></figure>`).join("") + "</div>" +
    `<p class="hint">Mosaic, scale and colour jitter are already applied: this is the ` +
    `tensor, not the files on disk. Written by the trainer itself — if these stop ` +
    `changing between rounds while the run continues, the vehicle is not re-reading ` +
    `its shard.</p>`;
}

// -- the trainer's own pictures ------------------------------------------------

const state = { data: null, vid: null, group: "consumed" };

const GROUPS = [
  ["consumed", "batches as fed", "The images after mosaic, scaling and colour jitter — " +
    "this is the tensor the network saw, not the file on disk."],
  ["truth", "truth vs prediction", "Same frames, ground truth beside this vehicle's own " +
    "prediction. Note what it misses, not only what it scores."],
  ["quality", "where it fails", "Confusion, precision against recall, and this vehicle's " +
    "own curves. All local: none of this is the aggregate on the holdout."],
];

export async function loadConsumed(vid) {
  state.vid = vid;
  const host = $("consumedBody");
  if (!host) return;
  if (!state.data) {
    host.innerHTML = '<div class="skel-block" style="min-height:200px"></div>';
    try {
      state.data = await (await fetch("/api/train-artifacts")).json();
    } catch {
      host.innerHTML = empty("The server did not answer.", "Is <code>python -m pipeline.server</code> still running?");
      return;
    }
  }
  renderConsumed();
}

function tile(vid, f) {
  return `<figure class="shot"><img loading="lazy" ` +
    `src="/api/train-artifact/${encodeURIComponent(vid)}/${encodeURIComponent(f.name)}" ` +
    `alt="${esc(f.caption)}"><figcaption>${esc(f.name)} — ${esc(f.caption)}</figcaption></figure>`;
}

function renderConsumed() {
  const host = $("consumedBody");
  const d = state.data;
  if (!host || !d) return;
  const list = d.vehicles || [];
  if (!list.length) {
    host.innerHTML = empty("No fleet on disk.", "Run the <code>fleet</code> stage, then reload.");
    return;
  }
  const v = list.find(x => String(x.vid) === String(state.vid)) || list.find(x => x.files.length) || list[0];
  state.vid = v.vid;

  const picker = list.map(x =>
    `<button class="chip" data-cvid="${x.vid}" data-sel="${String(x.vid) === String(v.vid)}" ` +
    `${x.files.length ? "" : "disabled"} title="${x.files.length} artifacts">v${x.vid}</button>`).join("");
  const tabs = GROUPS.map(([k, t]) =>
    `<button class="chip" data-cgroup="${k}" data-sel="${k === state.group}">${t}</button>`).join("");

  let body;
  if (!v.files.length) {
    body = empty(`Vehicle ${v.vid} has not trained yet.`,
      "These pictures appear after that vehicle's first round.");
  } else if (state.group === "truth") {
    // Paired on purpose: a prediction image alone is a demo, the two together are
    // evidence. Only pairs where both halves exist are shown.
    const rows = (d.pairs || []).filter(([a, b]) =>
      v.files.some(f => f.name === a) && v.files.some(f => f.name === b));
    body = rows.length
      ? '<div class="pairs">' + rows.map(([a, b]) =>
          `<div class="pair"><div><h3>ground truth</h3>${tile(v.vid, {name: a, caption: "ground truth"})}</div>` +
          `<div><h3>predicted</h3>${tile(v.vid, {name: b, caption: "predicted"})}</div></div>`).join("") + "</div>"
      : empty("No validation images from that round.", "They appear once a round completes validation.");
  } else {
    const files = v.files.filter(f => f.group === state.group);
    body = files.length
      ? '<div class="gallery">' + files.map(f => tile(v.vid, f)).join("") + "</div>"
      : empty("Nothing in this group yet.", "The last round wrote no such file.");
  }

  const blurb = (GROUPS.find(g => g[0] === state.group) || [])[2] || "";
  host.innerHTML =
    `<div class="chiprow">${picker}</div><div class="chiprow">${tabs}</div>` +
    `<p class="hint">${esc(blurb)}</p>` + body +
    `<p class="hint">Written by ultralytics itself, not by this project. The client passes ` +
    `<code>exist_ok=True</code>, so each vehicle's directory holds only its <b>last</b> ` +
    `round — this is a snapshot, never a history.</p>`;

  host.querySelectorAll("[data-cvid]").forEach(b => b.onclick = () => {
    state.vid = b.dataset.cvid; renderConsumed();
  });
  host.querySelectorAll("[data-cgroup]").forEach(b => b.onclick = () => {
    state.group = b.dataset.cgroup; renderConsumed();
  });
}
