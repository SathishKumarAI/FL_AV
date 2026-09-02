// The live batch feed, against a fake DOM.
//
// Three states, and the boring ones are the ones that go wrong: idle must explain
// itself rather than render an empty box, a vehicle that has started but not yet
// written a mosaic must say so rather than show the previous vehicle's, and the
// image URL must carry the mtime or the browser serves the last round's picture
// from cache for the whole run.
globalThis.document = { getElementById: () => null, querySelectorAll: () => [] };
globalThis.performance = { now: () => 0 };

const LISTING = {
  pairs: [],
  vehicles: [
    { vid: 3, condition: "rain / fog", dir: "runs/fl/batch3", files: [
      { name: "train_batch0.jpg", group: "consumed", caption: "a batch", mtime: 1786020949 },
      { name: "train_batch1.jpg", group: "consumed", caption: "a batch", mtime: 1786020950 },
      { name: "labels.jpg", group: "consumed", caption: "class counts", mtime: 1786020951 },
      { name: "val_batch0_pred.jpg", group: "pred", caption: "predicted", mtime: 1786020952 },
    ]},
    { vid: 4, condition: "highway", dir: null, files: [] },
  ],
};
globalThis.fetch = async () => ({ json: async () => LISTING });

const { renderNowTraining } = await import("./consumed.js");

// idle
const idle = { innerHTML: "" };
await renderNowTraining(idle, null, "idle", false);
console.assert(!idle.innerHTML.includes("<img"), "rendered an image while idle");
console.assert(idle.innerHTML.includes("Nothing is training"), "idle state is silent");

// training, with mosaics on disk
const live = { innerHTML: "" };
await renderNowTraining(live, 3, "vehicle 3 · rain / fog", true);
const imgs = live.innerHTML.match(/<img src="([^"]+)"/g) || [];
console.assert(imgs.length === 2, `expected the 2 train_batch files, got ${imgs.length}`);
console.assert(!live.innerHTML.includes("labels.jpg"), "labels.jpg is not a batch mosaic");
console.assert(!live.innerHTML.includes("val_batch0_pred"), "a prediction is not what it consumed");
console.assert(live.innerHTML.includes("train_batch0.jpg?t=1786020949"),
  "the mtime is missing from the URL, so the browser will serve a stale round");

// started, nothing written yet
const early = { innerHTML: "" };
await renderNowTraining(early, 4, "vehicle 4 · highway", true);
console.assert(!early.innerHTML.includes("<img"), "showed an image for a vehicle with none");
console.assert(early.innerHTML.includes("not written its"), "no explanation for the gap");

// a vehicle the listing has never heard of behaves like the one above, not like a crash
const unknown = { innerHTML: "" };
await renderNowTraining(unknown, 99, "vehicle 99", true);
console.assert(!unknown.innerHTML.includes("<img"), "invented an image for an unknown vehicle");

console.log("live feed: idle, training, not-yet-written and unknown vehicle all OK");
