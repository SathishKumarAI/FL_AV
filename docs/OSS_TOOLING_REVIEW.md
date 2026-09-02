# Open-source tooling: what to add, what to refuse

Reviewed 2026-08-16. The question was "what open-source projects should this one
assemble from?", and the honest answer for most of them is *none, because the thing
they provide is already on disk here*. That is rule 4 — assemble before building — read
in both directions: use what exists, and do not adopt a service to do what a file does.

Each row below is judged against three constraints this repo does not bend:

1. **Data is never committed** and never leaves the machine. That kills every hosted
   tracker regardless of quality.
2. **No credentials.** Nothing here needs any; nothing here may start needing any.
3. **`pipeline/` invokes `my-project/`, it does not modify it.** A tool that requires
   instrumentation inside the training loop costs a gated branch, not an afternoon.

## Verdicts

| Tool | What it would give | Verdict |
|---|---|---|
| **ultralytics' own output** | batch mosaics, label distributions, truth-vs-prediction, confusion matrix, PR/F1 curves — 14 files per vehicle per round | **wired.** They were already being written and never served. See `pipeline/train_artifacts.py` |
| **MLflow** | metrics store, run history, comparison | **already chosen, still broken.** It refused the last run: *"the filesystem tracking backend is in maintenance mode"*. Needs a SQLite backing store. Backlog 80 — fixing it is smaller than any panel that would duplicate it |
| **Ray Dashboard** | actor scheduling, GPU internals | **already assembled.** Linked from the header |
| **FiftyOne** (voxel51) | dataset curation app, embedding-based near-duplicate and label-error search, YOLO-format import | **deferred, not refused.** Its unique capability is embeddings, which is a phase-4 data-quality question. Its cost is a MongoDB process, a second port and a large install — too much to pay for *looking at pictures*, which is now free. Revisit for backlog 44 (data-quality audit) and 65 (leakage), where nothing else can do the job |
| **`supervision`** (Roboflow) | server-side box/mask drawing, YOLO↔COCO↔VOC conversion | **refused for the overlay.** YOLO labels are already normalised, so an SVG over the `<img>` draws them with no dependency and lets the browser toggle them without a refetch. Reconsider only if a *rendered* artifact is ever needed outside the browser |
| **Rerun** | live multimodal streaming viewer, excellent for watching a pipeline run | **refused for now.** Would require logging from `client_app.py` — the ⚠ zone — and the data here is per-round, not per-frame time series. Genuinely the right answer if per-step training telemetry is ever wanted |
| **DVC** | dataset versioning | **refused.** Already argued in `docs/PHASED_PLAN.md` phase 4: it adds a remote, a cache and a daemon to a repo whose hard rule is that data is never committed. The content-hash manifest is the part that was actually needed |
| **W&B / Comet / Neptune** | hosted experiment tracking | **refused on the credentials rule**, not on quality |
| **Label Studio / CVAT** | annotation | **not needed.** BDD100K arrives labelled; this project does not annotate |
| **Streamlit / Gradio** | a UI without writing one | **refused.** The dashboard exists, has no build step and serves off disk. Adding a framework now would be a rewrite disguised as an integration |
| **Opacus / Flower's DP mods** | differential privacy with a stated ε | **phase 5, item 8.** The honest privacy story, and the only one that makes "federated" mean more than "not pooled" |
| **Docker / uv lockfile** | reproducible environment | **accepted, separate branch.** See the production-readiness plan |

## The one rule this review kept running into

Every candidate that lost, lost the same way: it would have produced something the repo
already had, and charged a daemon for it. The winner was not a project at all — it was
noticing that the trainer had been drawing the pictures all along.

## Sources

- [voxel51/fiftyone](https://github.com/voxel51/fiftyone) — dataset curation, MongoDB backend
- [FiftyOne integrations](https://docs.voxel51.com/integrations/index.html)
- [rerun-io/rerun](https://github.com/rerun-io/rerun), [rerun-sdk on PyPI](https://pypi.org/project/rerun-sdk/) — 0.34.1, July 2026
- [supervision — annotators](https://supervision.roboflow.com/annotators/), [datasets](https://supervision.roboflow.com/datasets/)
