"""Detection metrics for time-series anomaly detection.

All functions are pure NumPy so they can be unit-tested without TensorFlow.

Conventions
-----------
* ``labels`` and ``predictions`` are 1-D binary arrays of equal length (one
  entry per timestep).
* An *event* (ground-truth anomaly) is a maximal run of consecutive ``1``
  labels. An *alert run* is a maximal run of consecutive ``1`` predictions.
* Point adjustment is applied per series (per channel) **before** pooling, so
  anomalies from neighbouring channels are never merged into one event.
* Undefined ratios (e.g. precision with no predictions) are reported as 0.0,
  matching ``sklearn``'s ``zero_division=0``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def _as_binary(values: Iterable) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return (array != 0).astype(np.int8)


def _check_lengths(labels: np.ndarray, predictions: np.ndarray) -> None:
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must have the same length ({len(labels)} != {len(predictions)})."
        )


def binary_runs(values: Iterable) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index pairs for every maximal run of ones."""
    binary = _as_binary(values)
    if binary.size == 0:
        return []
    padded = np.concatenate(([0], binary, [0]))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in zip(changes[0::2], changes[1::2])]


def anomaly_segments(labels: Iterable) -> list[tuple[int, int]]:
    """Ground-truth anomaly events as ``[start, end)`` pairs."""
    return binary_runs(labels)


def point_adjust(labels: Iterable, predictions: Iterable) -> np.ndarray:
    """Point-adjusted predictions.

    If any timestep inside a ground-truth event is predicted anomalous, the whole
    event is marked as detected. Predictions outside events are left unchanged.
    """
    labels = _as_binary(labels)
    adjusted = _as_binary(predictions).copy()
    _check_lengths(labels, adjusted)
    for start, end in anomaly_segments(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def confusion_counts(labels: Iterable, predictions: Iterable) -> dict[str, int]:
    labels = _as_binary(labels)
    predictions = _as_binary(predictions)
    _check_lengths(labels, predictions)
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
    }


def prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def strict_prf(labels: Iterable, predictions: Iterable) -> dict[str, float]:
    """Strict point-wise precision / recall / F1."""
    counts = confusion_counts(labels, predictions)
    return prf_from_counts(counts["tp"], counts["fp"], counts["fn"])


def adjusted_prf(labels: Iterable, predictions: Iterable) -> dict[str, float]:
    """Point-adjusted precision / recall / F1."""
    return strict_prf(labels, point_adjust(labels, predictions))


def detection_latencies(labels: Iterable, predictions: Iterable) -> list[int | None]:
    """Per ground-truth event: timesteps from event start to the first alert.

    Only alerts inside the event count (an alert at the first anomalous timestep
    has latency 0). Events with no alert inside them get ``None``.
    """
    labels = _as_binary(labels)
    predictions = _as_binary(predictions)
    _check_lengths(labels, predictions)
    latencies: list[int | None] = []
    for start, end in anomaly_segments(labels):
        hits = np.flatnonzero(predictions[start:end])
        latencies.append(int(hits[0]) if hits.size else None)
    return latencies


def mean_detection_latency(labels: Iterable, predictions: Iterable) -> float:
    """Mean latency over *detected* events; NaN if no event was detected."""
    detected = [lat for lat in detection_latencies(labels, predictions) if lat is not None]
    return float(np.mean(detected)) if detected else float("nan")


def count_false_alarms(labels: Iterable, predictions: Iterable) -> int:
    """Number of alert runs that do not overlap any ground-truth event."""
    labels = _as_binary(labels)
    predictions = _as_binary(predictions)
    _check_lengths(labels, predictions)
    return sum(1 for start, end in binary_runs(predictions) if not labels[start:end].any())


def false_alarms_per_1k(labels: Iterable, predictions: Iterable) -> float:
    """False alarm runs per 1,000 evaluated timesteps."""
    labels = _as_binary(labels)
    if labels.size == 0:
        return float("nan")
    return 1000.0 * count_false_alarms(labels, predictions) / labels.size


def safe_roc_auc(labels: Iterable, scores: Iterable) -> float:
    labels = _as_binary(labels)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_series(
    labels: Iterable,
    predictions: Iterable,
    scores: Iterable | None = None,
) -> dict[str, float | int | list]:
    """All metrics for one series (channel), plus the raw counts needed for pooling."""
    labels = _as_binary(labels)
    predictions = _as_binary(predictions)
    _check_lengths(labels, predictions)

    strict_counts = confusion_counts(labels, predictions)
    adjusted_counts = confusion_counts(labels, point_adjust(labels, predictions))
    strict = prf_from_counts(strict_counts["tp"], strict_counts["fp"], strict_counts["fn"])
    adjusted = prf_from_counts(adjusted_counts["tp"], adjusted_counts["fp"], adjusted_counts["fn"])
    latencies = detection_latencies(labels, predictions)
    detected = [lat for lat in latencies if lat is not None]
    false_alarms = count_false_alarms(labels, predictions)

    return {
        "precision": strict["precision"],
        "recall": strict["recall"],
        "f1": strict["f1"],
        "adjusted_precision": adjusted["precision"],
        "adjusted_recall": adjusted["recall"],
        "adjusted_f1": adjusted["f1"],
        "roc_auc": safe_roc_auc(labels, scores) if scores is not None else float("nan"),
        "events": len(latencies),
        "detected_events": len(detected),
        "event_recall": len(detected) / len(latencies) if latencies else float("nan"),
        "mean_latency": float(np.mean(detected)) if detected else float("nan"),
        "false_alarms": int(false_alarms),
        "false_alarms_per_1k": 1000.0 * false_alarms / labels.size if labels.size else float("nan"),
        "points": int(labels.size),
        "anomaly_points": int(labels.sum()),
        "predicted_anomalies": int(predictions.sum()),
        "strict_counts": strict_counts,
        "adjusted_counts": adjusted_counts,
        "latencies": detected,
    }


CHANNEL_MEAN_KEYS = (
    "precision",
    "recall",
    "f1",
    "adjusted_precision",
    "adjusted_recall",
    "adjusted_f1",
    "roc_auc",
    "event_recall",
    "mean_latency",
    "false_alarms_per_1k",
)


def aggregate_series_results(results: list[dict]) -> dict[str, dict[str, float]]:
    """Combine per-channel ``evaluate_series`` outputs.

    * ``channel_mean``: unweighted mean of each per-channel metric (NaNs ignored,
      e.g. latency for channels where nothing was detected).
    * ``pooled``: metrics recomputed from summed counts over all points. Point
      adjustment has already been applied per channel. Pooled latency is the mean
      over all detected events across channels.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty list of results.")

    channel_mean: dict[str, float] = {}
    for key in CHANNEL_MEAN_KEYS:
        values = np.array([r[key] for r in results], dtype=float)
        channel_mean[key] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")

    def summed(counts_key: str) -> dict[str, int]:
        return {k: int(sum(r[counts_key][k] for r in results)) for k in ("tp", "fp", "fn", "tn")}

    strict_counts = summed("strict_counts")
    adjusted_counts = summed("adjusted_counts")
    strict = prf_from_counts(strict_counts["tp"], strict_counts["fp"], strict_counts["fn"])
    adjusted = prf_from_counts(adjusted_counts["tp"], adjusted_counts["fp"], adjusted_counts["fn"])
    all_latencies = [lat for r in results for lat in r["latencies"]]
    events = int(sum(r["events"] for r in results))
    detected_events = int(sum(r["detected_events"] for r in results))
    points = int(sum(r["points"] for r in results))
    false_alarms = int(sum(r["false_alarms"] for r in results))

    pooled = {
        "precision": strict["precision"],
        "recall": strict["recall"],
        "f1": strict["f1"],
        "adjusted_precision": adjusted["precision"],
        "adjusted_recall": adjusted["recall"],
        "adjusted_f1": adjusted["f1"],
        "events": events,
        "detected_events": detected_events,
        "event_recall": detected_events / events if events else float("nan"),
        "mean_latency": float(np.mean(all_latencies)) if all_latencies else float("nan"),
        "median_latency": float(np.median(all_latencies)) if all_latencies else float("nan"),
        "false_alarms": false_alarms,
        "false_alarms_per_1k": 1000.0 * false_alarms / points if points else float("nan"),
        "evaluated_points": points,
        "anomaly_points": int(sum(r["anomaly_points"] for r in results)),
        "predicted_anomalies": int(sum(r["predicted_anomalies"] for r in results)),
    }
    return {"channel_mean": channel_mean, "pooled": pooled}
