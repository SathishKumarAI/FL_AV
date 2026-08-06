# Prompt — premium UI pass on the fleet dashboard

Written before the code. Backlog items 1–5, 12, 13, 22.

## The problem

The dashboard is correct and unreadable. Every panel is the same weight, so nothing
says which number matters; the charts are polylines with no axes, so a value can be
seen but not read; there are two separate places showing the same vehicle (`#smalls`
for learning, `#fleetCards` for shard info); and before the first poll lands every
figure is an em dash, which is indistinguishable from a broken server.

The single most important signal in this project — whether the aggregate checksum
moves between rounds — is drawn in the tenth panel down.

## What to build

| # | Item | Acceptance |
|---|---|---|
| 1 | Design system: type scale, 8px spacing grid, radii, one accent | no ad-hoc inline styling left in the panels; semantic colour used only for state |
| 2 | Real chart component: axes, gridlines, ticks, hover tooltip | inline SVG, no CDN, no build step; hovering any chart reads the value at that round |
| 3 | Fleet as one grid of vehicle channels: condition glyph, sparkline, delta chip, status ring | one card per vehicle, replacing both existing lists |
| 4 | Skeleton loaders instead of `—` before the first poll | a fresh page shows shimmering placeholders, not dashes |
| 5 | Empty states that name the next action | "no fleet yet" says which stage builds one |
| 12 | Per-vehicle detail drawer | full mAP curve, loss curves, weight movement, shard composition |
| 13 | Sample image strip per vehicle | see what "rain / fog" actually looks like, served from the vehicle's own shard |
| 22 | Accessibility | focus rings, ARIA on charts, contrast ≥ 4.5:1, reduced motion, Esc closes the drawer, cards reachable by keyboard |

## Design direction

**Night instrument cluster.** The subject is a fleet of vehicles, so the vernacular is
the one from a car at night: an instrument backlight (cyan) for data, and signal-lamp
semantics reserved strictly for state — amber caution, red fault, green go. Numerals
are set in tabular mono with the unit dimmed after them, the way a readout is.

The signature element is the **heartbeat band**: the round-over-round aggregate
checksum, drawn full width at the top of the live view with a lamp reading MOVING or
STUCK. It is the number this project's history says is worth more than all the others,
so it gets the position and the size that says so.

Not chosen: a bespoke charting library (rule 4 — assemble before building; these are
four line charts, inline SVG is enough), a build step (the page must be servable by
`http.server` from disk), a font download (no network at runtime — system stacks only).

## Supporting server work

The drawer needs two things the API does not expose:

- `GET /api/vehicle/<vid>` — shard composition: weather / scene / timeofday counts for
  the images that vehicle actually holds, from the cached BDD attribute index, plus a
  handful of sample image names. Must never trigger an index build — streaming 1.45 GB
  of JSON inside a request handler would hang the page.
- `GET /api/shard-image/<vid>/<name>` — the image bytes, from that vehicle's shard.

Both go through the same path guard already protecting `/reports/`, extracted so there
is one implementation rather than two.

## Non-negotiable

- `pipeline/` still never writes into `my-project/`.
- Stdlib only, loopback only, no build step, no CDN.
- No new data committed; images are served from shards that are already gitignored.

## Verification

- `python -m pytest pipeline/tests -q` — existing tests stay green, new ones cover the
  composition counts and the traversal guard on the image route.
- The page rendered in a real browser at 1280px, screenshotted, with a run's data in it.
