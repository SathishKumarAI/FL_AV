#!/usr/bin/env bash
# One command, clean checkout to results. Linux/macOS.
#
# Every step is the same command a person would type, in the same order, so this
# script is a runbook that happens to execute. Nothing here is a second code path.
#
#   ./scripts/run_pipeline.sh                       # demo profile, minutes
#   PROFILE=full ROUNDS=6 EPOCHS=4 PER_VEHICLE=1400 BASELINE=1 ./scripts/run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
PROFILE="${PROFILE:-demo}"
VEHICLES="${VEHICLES:-6}"
ROUNDS="${ROUNDS:-2}"
EPOCHS="${EPOCHS:-1}"
PER_VEHICLE="${PER_VEHICLE:-0}"
PARTITION="${PARTITION:-condition}"
ALPHA="${ALPHA:-0.5}"
STRATEGY="${STRATEGY:-fedavg}"
HOLDOUT_SIZE="${HOLDOUT_SIZE:-1000}"
BASELINE="${BASELINE:-0}"

# Without this flwr builds its own runtime env, installs the CPU-only torch wheel,
# and every client trains on CPU at ~5.5x the wall clock with no error anywhere.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

step() {
  echo
  echo "=== $1 ==="
  shift
  if ! "$@"; then
    echo "FAILED. Nothing downstream is run: a stage that fails halts the chain," >&2
    echo "because continuing past a failure is how this project used to ship" >&2
    echo "silent no-ops." >&2
    exit 1
  fi
}

step "Tests (fast, catches a broken checkout before the GPU is touched)" \
  "$PYTHON" -m pytest pipeline/tests my-project/tests -q

step "Shared holdout ($HOLDOUT_SIZE images no vehicle may see)" \
  "$PYTHON" -m pipeline.holdout --build --size "$HOLDOUT_SIZE" --seed 0

run_args=(-m pipeline.runner --all --profile "$PROFILE" --vehicles "$VEHICLES"
          --rounds "$ROUNDS" --epochs "$EPOCHS" --partition "$PARTITION"
          --alpha "$ALPHA" --strategy "$STRATEGY" --yes)
[ "$PER_VEHICLE" -gt 0 ] && run_args+=(--per-vehicle "$PER_VEHICLE")

step "Full chain: dataset, shards, fleet, validate, federate, evaluate, verify" \
  "$PYTHON" "${run_args[@]}"

if [ "$BASELINE" = "1" ]; then
  step "Centralised ceiling (pooled data, matched budget)" \
    "$PYTHON" -m pipeline.baseline --rounds "$ROUNDS" --local-epochs "$EPOCHS"
fi

step "Comparison against previous runs" "$PYTHON" -m pipeline.compare --last 5

echo
echo "Done."
echo "  report       : $(ls -dt pipeline/reports/*/ | head -1)report.html"
echo "  holdout curve: pipeline/.state/holdout_metrics.json"
echo "  dashboards   : $PYTHON -m pipeline.server  ->  http://127.0.0.1:8800"
