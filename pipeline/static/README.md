# The dashboard, by file

Open the one file that owns the thing you are changing. Nothing here needs a build
step: the server reads these files off disk on every request, so an edit is live on
reload. ES modules, no bundler, no CDN, no network at runtime.

| Change | File | Size |
|---|---|---|
| Anything visual — colour, spacing, type, a component's look | `app.css` | tokens at the top, components below |
| Page structure, a new panel, an element's `id` | `index.html` | markup only, no styles, no script |
| Axes, ticks, tooltips, sparkline, progress ring | `js/chart.js` | the only file that draws SVG plots |
| The run form, launch/stop, the stage table | `js/control.js` | |
| Polling, heartbeat, GPU readouts, criteria, log stream | `js/live.js` | |
| The fleet grid, the comparison and divergence charts | `js/fleet.js` | |
| The per-vehicle drawer | `js/drawer.js` | |
| The Data tab: counts, mixes, the shard table | `js/data.js` | |
| Label boxes over a frame, and the trainer's own pictures | `js/consumed.js` | the only file that draws over an image |
| Helpers, colours, condition glyphs | `js/util.js` | |
| What the views share | `js/state.js` | one object, documented per field |
| Wiring and startup | `js/main.js` | ~20 lines; if it grows, something is in the wrong file |

## Rules that keep it that way

1. **`index.html` holds no CSS and no JavaScript.** A style belongs in `app.css`, a
   behaviour in `js/`. Layout-only inline `style` on a grid container is the single
   exception, and it is still a smell.
2. **Colour is a token.** `var(--accent)`, never a hex literal, except in the chart
   palette in `util.js` where a series needs a stable per-index colour.
3. **One concern per module, and modules import downward:** `main` → views
   (`control`, `live`, `fleet`, `drawer`) → primitives (`chart`, `util`, `state`).
   `fleet` and `drawer` reference each other on purpose — ES module live bindings
   handle the cycle, because both are called only after load.
4. **Everything the page shows comes from `/api/state`,** which the server derives
   from files on disk. That is why a run launched from the CLI lights up the same
   panels as one launched from the form. The event stream is a latency optimisation,
   never the only source of a value.
5. **A new condition profile touches two files** — `PROFILES` in
   `pipeline/vehicles.py` and `GLYPHS` in `js/util.js`. Add both in the same commit.

## Server routes it depends on

| Route | Serves |
|---|---|
| `GET /` | `index.html` |
| `GET /static/...` | this directory, guarded against traversal |
| `GET /api/state` | everything the panels render |
| `GET /api/events` | SSE: log lines, stage transitions, signals |
| `GET /api/vehicle/<vid>` | shard composition and sample image names |
| `GET /api/shard-image/<vid>/<name>` | one image out of that vehicle's shard |
| `GET /api/shard-labels/<vid>/<name>` | that frame's label rows, normalised, for the overlay |
| `GET /api/train-artifacts` | which of the trainer's own pictures exist, per vehicle |
| `GET /api/train-artifact/<vid>/<name>` | one of them; `<name>` must be in `train_artifacts.KINDS` |
| `POST /api/run`, `POST /api/stop` | start and stop the one allowed run |

The label overlay is an SVG on the unit square laid over the same `<img>` — the frame
is never sent twice and nothing renders boxes server-side. That only works while the
image element is not cropped, which is why `.strip .ovfig img` overrides
`object-fit:cover`; `pipeline/tests/test_pipeline.py::test_label_overlay_boxes_land_where_the_label_file_says`
runs the conversion under node and fails if a box moves.
