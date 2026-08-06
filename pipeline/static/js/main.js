// Entry point: wires the tabs, the control form, the chart cursor, and starts
// the two data sources (polling for state, SSE for the log).
import { $ } from "./util.js";
import { enableChartCursor } from "./chart.js";
import { wireControl } from "./control.js";
import { poll, connectEvents, drawHeartbeat } from "./live.js";
import { renderFleet } from "./fleet.js";

function showView(tab) {
  document.querySelectorAll(".tab").forEach(x => x.setAttribute("aria-selected", String(x === tab)));
  $("view-control").hidden = tab.dataset.view !== "control";
  $("view-live").hidden = tab.dataset.view !== "live";
  // Charts sized while hidden come out wrong; redraw whatever just became visible.
  drawHeartbeat();
  renderFleet();
}
document.querySelectorAll(".tab").forEach(t => { t.onclick = () => showView(t); });

wireControl(() => showView(document.querySelector('.tab[data-view=live]')));
enableChartCursor();
connectEvents();
poll();
