"""Synthetic anomaly injection and root-cause attribution scoring.

Pure NumPy (no TensorFlow) so the logic is unit-testable.

Magnitudes are expressed in units of the target column's training standard
deviation (in the scaled space), with a floor so near-constant columns still get
a visible perturbation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

ANOMALY_TYPES = ("spike", "level_shift", "drift")

# (min, max) ranges used by ``sample_injections``.
SPIKE_SIGMAS = (6.0, 10.0)
LEVEL_SHIFT_SIGMAS = (3.0, 6.0)
LEVEL_SHIFT_LENGTH = (30, 150)
DRIFT_SIGMAS = (4.0, 8.0)
DRIFT_LENGTH = (100, 300)
SIGMA_FLOOR = 0.05


@dataclass(frozen=True)
class InjectionSpec:
    channel_id: str
    kind: str
    column: int
    start: int
    length: int
    magnitude: float  # signed offset in scaled units (peak offset for drift)

    @property
    def end(self) -> int:
        return self.start + self.length

    def to_dict(self) -> dict:
        return asdict(self)


def inject_anomaly(values: np.ndarray, spec: InjectionSpec) -> np.ndarray:
    """Return a copy of ``values`` (T x F) with the anomaly applied.

    * spike: ``magnitude`` added to ``spec.length`` timesteps (usually 1).
    * level_shift: constant ``magnitude`` added over the window.
    * drift: linear ramp from ~0 up to ``magnitude`` over the window.
    """
    if spec.kind not in ANOMALY_TYPES:
        raise ValueError(f"Unknown anomaly type {spec.kind!r}; expected one of {ANOMALY_TYPES}.")
    if spec.length < 1:
        raise ValueError("Injection length must be >= 1.")
    if spec.start < 0 or spec.end > len(values):
        raise ValueError(f"Injection [{spec.start}, {spec.end}) is outside a series of length {len(values)}.")
    if not 0 <= spec.column < values.shape[1]:
        raise ValueError(f"Column {spec.column} is outside {values.shape[1]} features.")

    injected = np.array(values, dtype=float, copy=True)
    if spec.kind == "drift":
        offset = spec.magnitude * np.arange(1, spec.length + 1) / spec.length
    else:
        offset = np.full(spec.length, spec.magnitude)
    injected[spec.start : spec.end, spec.column] += offset
    return injected.astype(values.dtype if np.issubdtype(values.dtype, np.floating) else float)


def injection_labels(length: int, spec: InjectionSpec) -> np.ndarray:
    labels = np.zeros(length, dtype=int)
    labels[spec.start : spec.end] = 1
    return labels


def _overlaps(start: int, end: int, segments: Iterable[tuple[int, int]], margin: int) -> bool:
    return any(start < seg_end + margin and seg_start - margin < end for seg_start, seg_end in segments)


def sample_injections(
    channels: Sequence[dict],
    n_injections: int,
    rng: np.random.Generator,
    min_start: int,
    avoid_margin: int = 0,
    max_tries: int = 100,
) -> list[InjectionSpec]:
    """Sample ``n_injections`` anomalies spread across ``channels``.

    Each channel dict needs ``channel_id``, ``n_points``, ``train_std`` (per-column
    std of scaled training data) and optionally ``segments`` (true anomaly
    ``[start, end)`` pairs to avoid). Types are balanced round-robin; the target
    column is drawn uniformly among columns that vary in training data, so the
    model has a notion of normal behaviour for it. Channels with no varying
    columns are skipped.
    """
    eligible = [c for c in channels if np.any(np.asarray(c["train_std"]) > 0)]
    if not eligible:
        raise ValueError("No channel has a non-constant training column to inject into.")

    specs: list[InjectionSpec] = []
    kinds = [ANOMALY_TYPES[i % len(ANOMALY_TYPES)] for i in range(n_injections)]
    rng.shuffle(kinds)
    for kind in kinds:
        channel = eligible[int(rng.integers(len(eligible)))]
        train_std = np.asarray(channel["train_std"], dtype=float)
        candidates = np.flatnonzero(train_std > 0)
        column = int(rng.choice(candidates))
        sigma = max(float(train_std[column]), SIGMA_FLOOR)
        sign = 1.0 if rng.random() < 0.5 else -1.0

        if kind == "spike":
            length = 1
            magnitude = sign * rng.uniform(*SPIKE_SIGMAS) * sigma
        elif kind == "level_shift":
            length = int(rng.integers(LEVEL_SHIFT_LENGTH[0], LEVEL_SHIFT_LENGTH[1] + 1))
            magnitude = sign * rng.uniform(*LEVEL_SHIFT_SIGMAS) * sigma
        else:
            length = int(rng.integers(DRIFT_LENGTH[0], DRIFT_LENGTH[1] + 1))
            magnitude = sign * rng.uniform(*DRIFT_SIGMAS) * sigma

        n_points = int(channel["n_points"])
        length = min(length, max(1, n_points - min_start - 1))
        high = n_points - length
        if high <= min_start:
            raise ValueError(f"Channel {channel['channel_id']} is too short for an injection.")

        segments = channel.get("segments", [])
        start = int(rng.integers(min_start, high))
        for _ in range(max_tries):
            if not _overlaps(start, start + length, segments, avoid_margin):
                break
            start = int(rng.integers(min_start, high))

        specs.append(
            InjectionSpec(
                channel_id=str(channel["channel_id"]),
                kind=kind,
                column=column,
                start=start,
                length=length,
                magnitude=float(magnitude),
            )
        )
    return specs


def attribution_rank(feature_scores: Sequence[float], target: int) -> int:
    """1-based rank of ``target`` when features are sorted by descending score.

    Ties are resolved pessimistically (a tied feature counts as ranked above the
    target), so a constant score vector never yields a spurious hit.
    """
    scores = np.asarray(feature_scores, dtype=float)
    if not 0 <= target < scores.size:
        raise ValueError(f"target {target} outside {scores.size} features.")
    if np.isnan(scores[target]):
        return int(scores.size)
    others = np.delete(scores, target)
    return int(np.sum(others >= scores[target]) + 1)


def top_k_hit(feature_scores: Sequence[float], target: int, k: int) -> bool:
    return attribution_rank(feature_scores, target) <= k


def summarize_attribution(records: Sequence[dict], ks: Sequence[int] = (1, 3)) -> dict:
    """Top-k hit rates per anomaly type (and overall).

    Each record needs ``kind``, ``rank`` and ``n_features``; an optional boolean
    ``detected`` adds a detection-rate column. Chance level is the expected hit
    rate of a uniformly random ranking.
    """
    def block(rows: Sequence[dict]) -> dict:
        if not rows:
            return {"n": 0}
        ranks = np.array([r["rank"] for r in rows], dtype=float)
        n_features = np.array([r["n_features"] for r in rows], dtype=float)
        out: dict[str, float | int] = {"n": len(rows), "mean_rank": float(ranks.mean())}
        for k in ks:
            out[f"top{k}_hit_rate"] = float(np.mean(ranks <= k))
            out[f"top{k}_chance"] = float(np.mean(np.minimum(k, n_features) / n_features))
        if all("detected" in r for r in rows):
            out["detection_rate"] = float(np.mean([bool(r["detected"]) for r in rows]))
        return out

    summary = {kind: block([r for r in records if r["kind"] == kind]) for kind in ANOMALY_TYPES}
    summary["overall"] = block(list(records))
    return summary
