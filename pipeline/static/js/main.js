// Entry point: wires the tabs, the control form, the chart cursor, and starts
// the two data sources (polling for state, SSE for the log).
import { $ } from "./util.js";
import { enableChartCursor } from "./chart.js";
import { wireControl } from "./control.js";
import { poll, connectEvents, drawHeartbeat } from "./live.js";
import { renderFleet } from "./fleet.js";
import { loadData } from "./data.js";
import { loadPlan } from "./plan.js";

const VIEWS = ["control", "live", "data", "plan"];

function showView(tab) {
  const want = tab.dataset.view;
  document.querySelectorAll(".tab").forEach(x => x.setAttribute("aria-selected", String(x === tab)));
  VIEWS.forEach(v => { $("view-" + v).hidden = v !== want; });
  // Charts sized while hidden come out wrong; redraw whatever just became visible.
  drawHeartbeat();
  renderFleet();
  // Data and Plan are fetched on first sight rather than on every poll: one costs
  // seconds of label reading, the other is only interesting when someone looks.
  if (want === "data") loadData(false);
  if (want === "plan") loadPlan();
}
document.querySelectorAll(".tab").forEach(t => { t.onclick = () => showView(t); });

wireControl(() => showView(document.querySelector('.tab[data-view=live]')));
enableChartCursor();
connectEvents();
poll();
