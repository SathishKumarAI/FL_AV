#!/usr/bin/env bash
# One-time DVC setup to move large data/weights out of git tracking while keeping
# the tiny split-list fixtures (train.txt/val.txt/test.txt) that the unit tests
# rely on. NON-destructive to git history (no force-push); it only stops tracking
# the blobs going forward and creates .dvc pointers.
#
# Prereqs: pip install "dvc[s3]"   and an S3-compatible bucket.
# Usage:   bash scripts/setup_dvc.sh
# Then:    git add *.dvc .gitignore && git commit && dvc push
set -euo pipefail

cd "$(dirname "$0")/.."   # my-project/

command -v dvc >/dev/null || { echo "dvc not installed: pip install 'dvc[s3]'"; exit 1; }
[ -d .dvc ] || dvc init --subdir

# Large model weights -> DVC (pointers committed, blobs pushed to remote).
for f in models/yolov8s.pt yolov8n.pt models/dump_models/*.pt logs/yolov8s.pt; do
  [ -f "$f" ] && dvc add "$f"
done

# Per-shard images + labels -> DVC. We intentionally DO NOT dvc-add the whole
# batch dir so train.txt/val.txt/test.txt stay in git (test fixtures + cheap).
for d in batch/batch_*/images batch/batch_*/labels batch34/batch_*/images batch34/batch_*/labels; do
  [ -d "$d" ] && dvc add "$d"
done

# Remote (S3-compatible). Override via env before running.
REMOTE_URL="${DVC_REMOTE_URL:-s3://CHANGE-ME-bucket/fl_av}"
dvc remote add -d --force storage "$REMOTE_URL"
[ -n "${AWS_ENDPOINT_URL:-}" ] && dvc remote modify storage endpointurl "$AWS_ENDPOINT_URL"

echo
echo "Next:"
echo "  git add -A '*.dvc' .gitignore .dvc && git commit -m 'chore: track data/weights with DVC'"
echo "  dvc push          # upload blobs to $REMOTE_URL"
echo "On a fresh clone:   dvc pull"
