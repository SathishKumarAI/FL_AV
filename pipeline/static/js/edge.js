// The live edge fleet: real machines running the global model on a camera.
//
// The fleet grid in fleet.js is the SIMULATED vehicles that train. This is the other
// half — machines that run what training produced. The column that matters is the
// round each node is on: during a federation the fleet is visibly split across rounds,
// and boxes on screen belong to the round the node reports, not to whatever the server
// has finished aggregating.
import { $, esc } from "./util.js";

const POLL_MS = 2000;

export async function pollEdge() {
  try {
    render(await (await fetch("/api/nodes")).json());
  } catch {
    // The dashboard's own poll reports connection loss; a second red banner here
    // would only say the same thing twice.
  } finally {
    setTimeout(pollEdge, POLL_MS);
  }
}

function render(d) {
  const nodes = d.nodes || [];
  $("edgeSummary").textContent = nodes.length
    ? `${d.online} of ${d.total} online · ${d.fleet_fps} fps · ${d.detections} objects`
    : "none reporting";

  if (!nodes.length) {
    $("edgeNodes").innerHTML = "";
    $("edgeNote").innerHTML =
      "No edge node has reported. Start one per machine:<br>" +
      "<code>python -m pipeline.edge --id cam-1 --server http://&lt;this-host&gt;:8800</code><br>" +
      "No camera? <code>--source synthetic</code> exercises every other step. " +
      "For nodes on other machines the dashboard must be started with " +
      "<code>--host 0.0.0.0</code>, which it will warn you about — " +
      "<code>/api/run</code> starts training subprocesses and nothing here authenticates.";
    return;
  }

  // Cache-busted per poll: the frame route sends no-store, but a stale src attribute
  // would never re-request at all and the fleet would look frozen.
  const t = Date.now();
  $("edgeNodes").innerHTML = '<div class="edges">' + nodes.map(n => {
    const cls = n.online ? "ok" : "warn";
    const state = n.online ? "live" : `gone ${n.age_s}s`;
    const top = Object.entries(n.counts || {}).sort((a, b) => b[1] - a[1]).slice(0, 4);
    return `<figure class="edge${n.online ? "" : " off"}">` +
      (n.has_frame
        ? `<img src="/api/node-frame/${encodeURIComponent(n.id)}?t=${t}" alt="latest frame from ${esc(n.label)}">`
        : `<div class="noframe">no frame yet</div>`) +
      `<figcaption>` +
        `<b>${esc(n.label)}</b> <span class="${cls}">${state}</span>` +
        `<div class="k">${n.fps.toFixed(1)} fps · ${n.latency_ms.toFixed(0)} ms · ` +
          `${esc(n.device || "?")}</div>` +
        `<div class="k">${n.model_round == null
            ? "<span class='warn'>no model yet</span>"
            : `round ${n.model_round}`} · ${n.detections} objects</div>` +
        (top.length
          ? `<div class="k">${top.map(([c, v]) => `${esc(c)} ${v}`).join(" · ")}</div>`
          : "") +
        (n.error ? `<div class="k warn">${esc(n.error)}</div>` : "") +
      `</figcaption></figure>`;
  }).join("") + "</div>";

  // Nodes lag by design: each keeps running the model it downloaded until a newer one
  // is published. Saying so stops the split being read as a bug.
  const rounds = [...new Set(nodes.filter(n => n.online && n.model_round != null)
                                  .map(n => n.model_round))];
  $("edgeNote").innerHTML =
    `Each node runs the newest global checkpoint it has downloaded, and checks for a ` +
    `newer one periodically — so during a federation the fleet is split across rounds ` +
    `by design.` +
    (rounds.length > 1
      ? ` <b>Right now: rounds ${rounds.sort((a, b) => a - b).join(", ")}.</b> ` +
        `Boxes on each frame are that node's round, not the server's latest.`
      : rounds.length === 1
        ? ` All online nodes are on round <b>${rounds[0]}</b>.`
        : "") +
    ` Nothing here trains: a camera stream has no labels, and training on the model's ` +
    `own predictions is a confirmation loop. Training stays on the labelled shards.`;
}
