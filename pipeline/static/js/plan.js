// Plan view: what the configuration on the Control tab will actually do.
//
// The estimate used to be one line from one measured constant. It did not say how
// many image-visits the configuration implies, that a centralised ceiling needs the
// same number to be comparable, or what the equivalent command is. That arithmetic
// living in somebody's head is how a baseline shipped with 1.667x the budget.
import { $, esc, empty } from "./util.js";

let current = null;

export async function loadPlan() {
  $("planBody").innerHTML = '<div class="skel-block" style="min-height:240px"></div>';
  try {
    current = await (await fetch("/api/plan")).json();
  } catch {
    $("planBody").innerHTML = empty("The server did not answer.", "Is it still running?");
    return;
  }
  render();
}

const mins = (s) => s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;

function render() {
  const p = current;
  if (!p) return;
  const b = p.budget, c = p.config;

  $("planBody").innerHTML =
    `<div class="panel"><h2>Budget<span class="n">what both sides of a comparison must agree on</span></h2>` +
      `<div class="sum">${b.vehicles} vehicles <em>×</em> ${b.images_per_vehicle.toLocaleString()} images ` +
      `<em>×</em> ${b.rounds} rounds <em>×</em> ${b.local_epochs} local epochs ` +
      `<em>=</em> <b>${b.image_visits.toLocaleString()}</b> image-visits</div>` +
      `<div class="readouts" style="margin-top:var(--s4)">` +
        `<div class="readout"><span class="v">${b.effective_epochs}</span>` +
          `<div class="k">effective epochs per vehicle</div></div>` +
        `<div class="readout"><span class="v">${mins(b.seconds_estimate)}</span>` +
          `<div class="k">estimated GPU time</div></div>` +
        `<div class="readout"><span class="v">${b.wh_estimate}</span><span class="u">Wh</span>` +
          `<div class="k">estimated energy</div></div>` +
        `<div class="readout"><span class="v">${b.imgsz}</span><span class="u">px</span>` +
          `<div class="k">image size</div></div>` +
        `<div class="readout"><span class="v">${b.pooled_images.toLocaleString()}</span>` +
          `<div class="k">pooled images for the ceiling</div></div>` +
        `<div class="readout"><span class="v">${b.centralised_epochs_to_match}</span>` +
          `<div class="k">epochs to match it</div></div>` +
      `</div>` +
      `<p class="hint">A centralised model trained on those ${b.pooled_images.toLocaleString()} ` +
      `images for ${b.centralised_epochs_to_match} epochs makes exactly the same ` +
      `${b.image_visits.toLocaleString()} image-visits. Give it more and the gap measures ` +
      `the budget rather than the method — which is what happened the first time this ` +
      `project ran one.</p>` +
      `<p class="hint">Estimates are calibrated on a measured run: 6 vehicles × 1 400 ` +
      `images × 6 rounds × 4 epochs took 3 296 s and 82.2 Wh on an RTX 5070 Ti.</p></div>` +

    (p.warnings && p.warnings.length
      ? `<div class="panel"><h2>Before you start</h2><ul class="plain">` +
        p.warnings.map(w => `<li><span class="lamp s-needs_confirm">check</span>` +
          `<span style="color:var(--ink-2)">${esc(w)}</span></li>`).join("") + `</ul></div>`
      : "") +

    `<div class="panel"><h2>Stages<span class="n">for this configuration</span></h2>` +
      `<table><thead><tr><th scope="col">Stage</th><th scope="col">State</th>` +
      `<th scope="col">Cost</th><th scope="col">Why</th></tr></thead><tbody>` +
      p.stages.map(s => {
        const cls = s.satisfied ? "s-skipped" : (s.gated ? "s-needs_confirm" : "s-pending");
        const word = s.satisfied ? "skip" : (s.gated ? "gated" : "will run");
        return `<tr><td><b>${esc(s.name)}</b><br><span style="color:var(--dim);` +
          `font-size:var(--t-sm)">${esc(s.title)}</span></td>` +
          `<td><span class="lamp ${cls}">${word}</span></td>` +
          `<td class="num">${esc(s.est)}</td>` +
          `<td style="color:var(--dim);font-size:var(--t-sm)">${esc(s.detail)}</td></tr>`;
      }).join("") + `</tbody></table></div>` +

    `<div class="panel"><h2>The same thing, from a terminal` +
      `<span class="n">no dashboard required</span></h2>` +
      p.commands.map((cmd, i) =>
        `<div class="cmd"><div class="lbl">${esc(cmd.label)}</div>` +
        `<pre class="cmdline" id="cmd${i}">${esc(cmd.cmd)}</pre>` +
        `<button class="copy" data-cmd="${i}">Copy</button></div>`).join("") +
      `<p class="hint">Every arm of a comparison changes exactly one setting and ends ` +
      `scored on the same holdout, which is the only number comparable between runs.</p></div>` +

    `<div class="panel"><h2>On disk right now</h2><dl class="kvs">` +
      `<dt>fleet partition</dt><dd>${esc((p.fleet_on_disk || {}).partition ?? "none")}</dd>` +
      `<dt>fleet seed</dt><dd>${esc(String((p.fleet_on_disk || {}).seed ?? "—"))}</dd>` +
      `<dt>images per vehicle</dt><dd>${esc(String((p.fleet_on_disk || {}).per_vehicle ?? "—"))}</dd>` +
      `<dt>fleet fingerprint</dt><dd>${esc((p.fleet_on_disk || {}).fingerprint ?? "—")}</dd>` +
      `<dt>holdout</dt><dd>${esc(String((p.holdout || {}).size ?? "none"))} images, seed ` +
      `${esc(String((p.holdout || {}).seed ?? "—"))}</dd>` +
      `<dt>strategy this run</dt><dd>${esc(c.strategy)}</dd>` +
      `<dt>partition this run</dt><dd>${esc(c.partition)}` +
      `${c.partition === "dirichlet" ? ` α=${c.alpha}` : ""}</dd>` +
      `</dl><p class="hint">If the fleet on disk does not match the configuration, the ` +
      `fleet stage rebuilds it rather than reusing shards that answer a different ` +
      `question.</p></div>`;

  document.querySelectorAll("#planBody .copy").forEach(b => b.onclick = async () => {
    const text = $("cmd" + b.dataset.cmd).textContent;
    try {
      await navigator.clipboard.writeText(text);
      b.textContent = "Copied";
    } catch {
      // Clipboard is blocked on http:// in some browsers; select it instead so the
      // keyboard can still take it.
      const range = document.createRange();
      range.selectNodeContents($("cmd" + b.dataset.cmd));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      b.textContent = "Selected";
    }
    setTimeout(() => { b.textContent = "Copy"; }, 1600);
  });
}
