// Docs view: what each program does, and which number it contributes to.
//
// The text comes from the modules' own docstrings, served by /api/docs, so this page
// cannot describe a module that has since changed. What it adds is the part a
// docstring cannot know: which of this project's numbers a program feeds, and which
// tab shows its output.
import { $, esc, empty } from "./util.js";

let index = null;
let filter = "";

export async function loadDocs() {
  if (index) return render();
  $("docsBody").innerHTML = '<div class="skel-block" style="min-height:260px"></div>';
  try {
    index = await (await fetch("/api/docs")).json();
  } catch {
    $("docsBody").innerHTML = empty("The server did not answer.", "Is it still running?");
    return;
  }
  render();
}

/** First paragraph of a docstring: enough to know whether to open the file. */
function lead(doc) {
  const para = (doc || "").split("\n\n").slice(0, 2).join("\n\n");
  return para.length > 420 ? para.slice(0, 420) + "…" : para;
}

function matches(m) {
  if (!filter) return true;
  const hay = `${m.module} ${m.summary} ${m.contributes} ${m.tab} ${m.command}`.toLowerCase();
  return hay.includes(filter);
}

function render() {
  const d = index;
  if (!d) return;
  const shown = d.modules.filter(matches);

  $("docsBody").innerHTML =
    `<div class="panel"><h2>How a run flows<span class="n">left to right, halting on the first failure</span></h2>` +
      `<div class="chain">` + d.chain.map(s =>
        `<span class="node${s.gated ? " gated" : ""}" title="${esc(s.title)}${s.gated ?
          " — needs confirmation, it costs real time or GPU" : ""}">${esc(s.name)}` +
        `${s.gated ? " <em>gate</em>" : ""}</span>`).join('<span class="arrow">→</span>') +
      `</div>` +
      `<p class="hint">A stage that fails stops the chain. Every stage runs as a ` +
      `subprocess with <code>cwd=my-project</code>, because flwr's detached SuperLink ` +
      `caches the working directory of whichever run started it — launching from ` +
      `anywhere else makes every relative path resolve somewhere wrong.</p></div>` +

    `<div class="panel"><h2>The tabs<span class="n">what each one is for</span></h2>` +
      `<table><thead><tr><th scope="col">Tab</th><th scope="col">Answers</th>` +
      `<th scope="col">Reads</th></tr></thead><tbody>` +
      d.tabs.map(t => `<tr><td><b>${esc(t.name)}</b></td>` +
        `<td>${esc(t.answers)}</td>` +
        `<td style="color:var(--dim);font-size:var(--t-sm)">${esc(t.reads)}</td></tr>`).join("") +
      `</tbody></table>` +
      `<p class="hint">Every number on every tab is read from files on disk rather ` +
      `than held in the server, which is why a run launched from a terminal lights up ` +
      `the same panels as one launched from the form.</p></div>` +

    `<div class="panel"><h2>Programs<span class="n">${shown.length} of ${d.modules.length}</span></h2>` +
      `<label class="check" style="margin-bottom:var(--s3)">` +
      `<input id="docsFilter" type="search" placeholder="filter by name, tab or what it does" ` +
      `value="${esc(filter)}" style="max-width:420px"></label>` +
      (shown.length ? `<div class="docgrid">` + shown.map(m =>
        `<article class="doccard">` +
          `<header><code>${esc(m.module)}</code>` +
          `<span class="tabtag" title="where its output appears">${esc(m.tab)}</span></header>` +
          `<p class="sumline">${esc(m.summary)}</p>` +
          `<p class="why"><b>Contributes:</b> ${esc(m.contributes)}</p>` +
          (m.command && m.command !== "—"
            ? `<pre class="cmdline">${esc(m.command)}</pre>` : "") +
          `<details><summary>the module's own docstring</summary>` +
          `<pre class="docstring">${esc(lead(m.doc))}</pre></details>` +
        `</article>`).join("") + `</div>`
        : empty("Nothing matches that.", "Clear the filter to see all programs.")) +
      `</div>` +

    `<div class="panel"><h2>Project documents<span class="n">${d.documents.length} files</span></h2>` +
      `<table><thead><tr><th scope="col">File</th><th scope="col">Title</th>` +
      `<th scope="col">Lines</th></tr></thead><tbody>` +
      d.documents.map(doc => `<tr><td><code>${esc(doc.path)}</code></td>` +
        `<td>${esc(doc.heading)}</td><td class="num">${doc.lines}</td></tr>`).join("") +
      `</tbody></table>` +
      `<p class="hint">Start with <code>docs/RUNBOOK.md</code> to run it, ` +
      `<code>STATUS.md</code> for where the last session stopped, and ` +
      `<code>CLAUDE.md</code> for the rules and the list of silent failures this ` +
      `project has already shipped.</p></div>`;

  const box = $("docsFilter");
  if (box) {
    box.oninput = () => {
      filter = box.value.trim().toLowerCase();
      const at = box.selectionStart;
      render();
      const next = $("docsFilter");
      if (next) { next.focus(); next.setSelectionRange(at, at); }
    };
  }
}
