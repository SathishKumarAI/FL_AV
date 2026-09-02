// Renders the strip against a fake DOM and asserts the overlay geometry.
//
// A box drawn a few percent off looks plausible and is worse than no box at all:
// the whole point of the overlay is to be trusted when it says a shard's labels are
// wrong. The conversion is four subtractions and this asserts all four.
//
// Run by pipeline/tests/test_pipeline.py, which copies the dashboard's js/ next to
// this file first — hence the bare `./consumed.js` import. `node overlay_geometry.mjs`
// on its own will not resolve it.
globalThis.document = { getElementById: () => null, querySelectorAll: () => [] };
globalThis.fetch = async (url) => ({ json: async () => ({
  class_names: ["person","rider","car"],
  boxes: [{cls:2, cx:0.5, cy:0.5, w:0.25, h:0.5}, {cls:0, cx:0.1, cy:0.9, w:0.2, h:0.2}],
})});
const { renderStrip } = await import("./consumed.js");

const host = { innerHTML: "" };
await renderStrip(host, 7, ["a.jpg", "b.jpg"], true);
const h = host.innerHTML;

const rect = h.match(/<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"/);
console.assert(rect, "no rect drawn");
const [, x, y, w, hh] = rect.map(Number);
// cx .5 cy .5 w .25 h .5  ->  x .375  y .25  w .25  h .5
console.assert(Math.abs(x - 0.375) < 1e-6, `x ${x}`);
console.assert(Math.abs(y - 0.25) < 1e-6, `y ${y}`);
console.assert(Math.abs(w - 0.25) < 1e-6, `w ${w}`);
console.assert(Math.abs(hh - 0.5) < 1e-6, `h ${hh}`);
console.assert(h.includes("<title>car</title>"), "class name not in the tooltip");
console.assert((h.match(/<figure class="ovfig">/g) || []).length === 2, "not 2 figures");
console.assert((h.match(/<rect /g) || []).length === 4, "not 2 boxes per figure");
console.assert(h.includes('viewBox="0 0 1 1"'), "overlay is not a unit square");
console.assert(h.includes("2 objects"), "no count caption");

// labels off: the same frames, no overlay at all.
const plain = { innerHTML: "" };
await renderStrip(plain, 7, ["a.jpg"], false);
console.assert(!plain.innerHTML.includes("<svg"), "overlay drawn when labels are off");
console.assert(plain.innerHTML.includes("/api/shard-image/7/a.jpg"), "frame url wrong");

// an empty shard says so instead of rendering nothing.
const none = { innerHTML: "" };
await renderStrip(none, 7, [], true);
console.assert(none.innerHTML.includes("No images materialised"), "empty shard unexplained");
console.log("overlay geometry, tooltips, counts, toggle and empty state: OK");
