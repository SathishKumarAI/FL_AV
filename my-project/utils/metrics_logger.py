"""Per-round federated metrics logging to CSV plus an end-of-run summary.

The Flower strategy calls into this module from ``aggregate_fit`` /
``aggregate_evaluate`` to persist a row per round/stage to ``logs/metrics.csv``
and to emit a best-mAP summary when the final round completes.
"""
import csv
import os
from pathlib import Path

from utils.logging_setup import configure_logging

logger = configure_logging("metrics", "logs/metrics.log")

# Metric keys reported by YOLO clients (see client_app.py).
METRIC_KEYS = ["precision", "recall", "mAP50", "mAP50-95", "fitness"]


def aggregate_client_metrics(results):
    """Weighted average of client metrics, weighted by ``num_examples``.

    Args:
        results: list of ``(client_proxy, res)`` where ``res`` has ``num_examples``
                 and a ``metrics`` dict.

    Returns:
        dict mapping each METRIC_KEYS entry to its weighted mean (0.0 if no data).
    """
    acc = {k: 0.0 for k in METRIC_KEYS}
    total = 0
    for _, res in results:
        n = int(getattr(res, "num_examples", 0) or 0)
        if n <= 0:
            n = 1
        metrics = getattr(res, "metrics", {}) or {}
        for k in METRIC_KEYS:
            try:
                acc[k] += float(metrics.get(k, 0.0)) * n
            except (TypeError, ValueError):
                pass
        total += n
    if total == 0:
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: acc[k] / total for k in acc}


class MetricsLogger:
    """Append per-round metrics to a CSV and log a final summary."""

    def __init__(self, csv_path="logs/metrics.csv"):
        self.csv_path = csv_path
        self.fieldnames = ["round", "stage", "num_clients", "loss"] + METRIC_KEYS
        self.history = []
        self._started = False
        # Deliberately no file I/O here. Constructing a strategy used to truncate
        # logs/metrics.csv, so anything that merely *built* one in the project
        # directory -- a test, a registry probe, an interactive check -- destroyed
        # the metrics of a run that was still going. It happened: a strategy probe
        # wiped rounds 1-5 of a six-round run in its final minute, and the run's own
        # verify then failed for want of rows it had written itself.
        logger.info(f"[Metrics] per-round metrics will be written to {self.csv_path}")

    def _start_file(self):
        """Truncate and write the header, once, on the first row of a real run."""
        Path(os.path.dirname(self.csv_path) or ".").mkdir(parents=True, exist_ok=True)
        # Fresh header each run so stale rows don't pile up across runs.
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        self._started = True
        logger.info(f"[Metrics] logging per-round metrics to {self.csv_path}")

    def log_round(self, server_round, stage, metrics, num_clients, loss=None):
        """Write one row for (round, stage) and remember it for the summary."""
        row = {
            "round": server_round,
            "stage": stage,
            "num_clients": num_clients,
            "loss": None if loss is None else round(float(loss), 5),
        }
        for k in METRIC_KEYS:
            try:
                row[k] = round(float(metrics.get(k, 0.0)), 5)
            except (TypeError, ValueError):
                row[k] = 0.0
        if not self._started:
            self._start_file()
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
        self.history.append(row)
        logger.info(f"[Metrics] round={server_round} stage={stage} -> {row}")

    def summary(self):
        """Log best-mAP50 across rounds (prefers evaluate rows)."""
        evals = [r for r in self.history if r["stage"] == "evaluate"]
        rows = evals or self.history
        if not rows:
            logger.info("[Metrics] no data collected; skipping summary.")
            return
        best = max(rows, key=lambda r: r.get("mAP50", 0.0))
        num_rounds = len({r["round"] for r in self.history})
        logger.info(
            f"[Metrics] === SUMMARY === rounds={num_rounds} | "
            f"best mAP50={best['mAP50']} @ round {best['round']} ({best['stage']}) | "
            f"precision={best['precision']} recall={best['recall']} "
            f"mAP50-95={best['mAP50-95']} | CSV: {self.csv_path}"
        )
