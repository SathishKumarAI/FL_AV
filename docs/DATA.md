# Data & Model Management

Why and how FL_AV handles its dataset shards and model weights — and how to stop
shipping ~400 MB of binaries inside the git repo.

## The problem

| Tracked in git | Size | Issue |
|----------------|------|-------|
| `batch/` (10 BDD100K shards) | ~326 MB | images/labels in version control bloat every clone |
| `models/*.pt`, `yolov8n.pt`, `models/dump_models/*.pt`, `logs/yolov8s.pt` | ~75 MB | binary weights + duplicates, no versioning |

Git stores every version of these forever — clones are slow, diffs are useless,
and there's no model lineage.

## The approach: DVC pointers + object storage

[DVC](https://dvc.org) replaces large files with tiny `*.dvc` pointer files
(committed to git) and stores the actual bytes in an S3-compatible remote.

**Crucially, the split-list files (`train.txt`/`val.txt`/`test.txt`) stay tracked
in git** — they're tiny, and the unit tests use them as fixtures
(`count_shard_examples`). Only `images/` + `labels/` dirs and `.pt` weights go to
DVC. `scripts/setup_dvc.sh` is written to preserve this split.

```bash
cd my-project
pip install "dvc[s3]"
export DVC_REMOTE_URL=s3://your-bucket/fl_av     # or set in .env
bash scripts/setup_dvc.sh
git add -A '*.dvc' .gitignore .dvc && git commit -m "chore: track data/weights with DVC"
dvc push                                          # upload blobs

# fresh clone / CI / a new SuperNode:
dvc pull
```

> **History note:** this is **non-destructive** — it stops tracking the blobs
> going forward but they remain in git *history*, so existing clones don't shrink.
> Reclaiming that space needs a one-time `git filter-repo` purge + force-push
> (destructive; all clones must re-clone). That step is intentionally **not**
> automated and is gated on an explicit decision — see
> [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) §7.

## Trained global models

Federated checkpoints land in `checkpoints/` (`global_round_N.pt`,
`global_last.pt`) — gitignored. Publish the ones you keep to a model
registry / object-store prefix rather than committing them; treat
`global_last.pt` as the deployable artifact.

## Duplicate weights to retire

`models/dump_models/` and `logs/yolov8s.pt` duplicate `models/yolov8s.pt`. Per the
archive-don't-delete policy, move them under `archive/` (or DVC-track then
git-untrack) rather than deleting. They are not referenced by the FL code
(`MODEL_PATH = "models/yolov8s.pt"`).

## Secrets / config

Runtime config and credentials come from the environment, never from
`pyproject.toml` or images. Copy `.env.example` → `.env` (gitignored):
`FL_AV_DATA_ROOT`, log tuning, SuperLink host, and DVC/S3 credentials.
