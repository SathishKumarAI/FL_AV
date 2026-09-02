// Control view: the run form, the launch/stop buttons, the stage table.
import { $, esc } from "./util.js";

export function config() {
  return {
    profile: $("profile").value,
    vehicles: +$("vehicles").value,
    rounds: +$("rounds").value,
    epochs: +$("epochs").value,
    seed: +$("seed").value,
    partition: $("partition").value,
    alpha: +$("alpha").value,
    strategy: $("strategy").value,
    proximal_mu: +$("mu").value,
    confirm: $("confirm").checked,
  };
}

export function estimate() {
  const c = config();
  const per = c.profile === "demo" ? 300 : 6308;
  // ~919 s measured for 2 clients x 2 rounds x 1 epoch on 6308 images, serialised.
  const secs = (per / 6308) * 230 * c.vehicles * c.rounds * c.epochs;
  const m = Math.max(1, Math.round(secs / 60));
  $("estimate").textContent =
    `${c.vehicles} vehicles × ${per} images × ${c.rounds} rounds × ${c.epochs} epochs — ` +
    `roughly ${m} min of GPU time.`;
  const dirichlet = c.partition === "dirichlet";
  $("alphaWrap").hidden = !dirichlet;
  $("alphaNote").hidden = !dirichlet;
  $("muWrap").hidden = c.strategy !== "fedprox";
}

/** Fill the strategy and partition menus from what the server actually registered,
 *  so a name the backend does not know cannot be picked here. */
export function renderOptions(options) {
  if (!options) return;
  const fill = (id, values) => {
    const el = $(id);
    if (!el || el.dataset.filled === String(values.length)) return;
    const keep = el.value;
    el.innerHTML = values.map(v => `<option value="${v}">${v}</option>`).join("");
    el.value = values.includes(keep) ? keep : values[0];
    el.dataset.filled = String(values.length);
  };
  fill("strategy", options.strategies || []);
  estimate();
}

export function renderStages(rows) {
  if (!rows) return;
  $("stageCount").textContent = `${rows.filter(r => !r.satisfied).length} of ${rows.length} will run`;
  $("stageTable").innerHTML = rows.map(r => {
    const cls = r.satisfied ? "s-skipped" : (r.gated ? "s-needs_confirm" : "s-pending");
    const word = r.satisfied ? "skip" : (r.gated ? "gated" : "will run");
    return `<tr><td><b>${esc(r.name)}</b><br>` +
      `<span style="color:var(--dim);font-size:var(--t-sm)">${esc(r.title)}</span></td>` +
      `<td><span class="lamp ${cls}">${word}</span></td>` +
      `<td class="num">${esc(r.est)}</td>` +
      `<td style="color:var(--dim);font-size:var(--t-sm)">${esc(r.detail)}</td></tr>`;
  }).join("");
}

export function wireControl(onLaunched) {
  ["profile", "vehicles", "rounds", "epochs", "partition", "strategy"].forEach(id => {
    $(id).oninput = estimate;
    $(id).onchange = estimate;
  });
  estimate();

  $("launch").onclick = async () => {
    $("launch").disabled = true;
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config()),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      $("estimate").innerHTML = `<span class="warn">${esc(body.error ||
        "A run is already in flight. Stop it before launching another.")}</span>`;
      $("launch").disabled = false;
      return;
    }
    onLaunched();
  };
  $("stop").onclick = () => fetch("/api/stop", { method: "POST" });
}
