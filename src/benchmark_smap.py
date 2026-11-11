"""Benchmark the LSTM autoencoder against baselines on NASA SMAP / MSL.

Per channel, three detectors are evaluated on the same test split:

* LSTM autoencoder (reconstruction error),
* Isolation Forest,
* a trivial first-difference baseline (``|x_t - x_{t-1}|`` on the telemetry column).

All three use the same thresholding rule: smooth the scores, then threshold at a
statistic (percentile or mean + k*std) of the smoothed *training* scores.

Reported metrics (see ``metrics.py`` for exact definitions): strict point-wise and
point-adjusted precision / recall / F1 as per-channel mean and pooled over all
points, ROC-AUC, mean detection latency and false alarms per 1,000 points.

Root-cause attribution is validated by injecting synthetic anomalies (spike,
level shift, gradual drift) into random columns / timesteps of the test data and
checking whether the injected column ranks top-1 / top-3 by per-feature
reconstruction error (see ``injection.py``).
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras

from baselines import first_difference_scores
from data_loader import build_point_labels, iter_telemanom_channels, load_numpy_array
from injection import (
    InjectionSpec,
    attribution_rank,
    inject_anomaly,
    sample_injections,
    summarize_attribution,
)
from metrics import aggregate_series_results, anomaly_segments, evaluate_series, safe_roc_auc
from model import build_lstm_autoencoder
from predict import reconstruction_scores
from preprocess import align_sequence_scores, create_sequences
from utils import calculate_threshold, ensure_parent, moving_average, save_json

MODEL_KEYS = {
    "ae": "lstm_autoencoder",
    "if": "isolation_forest",
    "diff": "first_difference",
}
MODEL_LABELS = {
    "ae": "LSTM-AE",
    "if": "Isolation Forest",
    "diff": "|x_t - x_{t-1}|",
}
ROW_METRIC_KEYS = (
    "precision",
    "recall",
    "f1",
    "adjusted_precision",
    "adjusted_recall",
    "adjusted_f1",
    "roc_auc",
    "threshold",
    "events",
    "detected_events",
    "mean_latency",
    "false_alarms",
    "false_alarms_per_1k",
)


def fit_autoencoder(
    train_values: np.ndarray,
    window_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[keras.Model, np.ndarray]:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    model = build_lstm_autoencoder(
        window_size=window_size,
        n_features=train_values.shape[1],
        latent_dim=latent_dim,
    )

    n_sequences = len(train_values) - window_size + 1
    if n_sequences <= 50000:
        sequences = create_sequences(train_values, window_size=window_size)
        callbacks: list[keras.callbacks.Callback] = []
        if validation_split > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                )
            )

        model.fit(
            sequences,
            sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )
        return model, sequences

    dataset = keras.utils.timeseries_dataset_from_array(
        train_values,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size,
    )
    dataset = dataset.map(lambda batch: (batch, batch))
    model.fit(dataset, epochs=epochs, verbose=0)
    return model, np.empty((0, window_size, train_values.shape[1]), dtype=np.float32)


def sequence_scores_stream(model: keras.Model, values: np.ndarray, window_size: int, batch_size: int) -> np.ndarray:
    dataset = keras.utils.timeseries_dataset_from_array(
        values,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )
    scores: list[np.ndarray] = []
    for batch in dataset:
        reconstructed = model(batch, training=False)
        batch_scores = tf.reduce_mean(tf.square(batch - reconstructed), axis=(1, 2)).numpy()
        scores.append(batch_scores)
    return np.concatenate(scores)


def window_feature_errors(model: keras.Model, values: np.ndarray, window_size: int) -> np.ndarray:
    """Per-window, per-feature reconstruction MSE, shape (n_windows, n_features).

    Window ``i`` covers ``values[i : i + window_size]``. This is the same quantity
    ``predict.py`` uses for root-cause attribution.
    """
    sequences = create_sequences(np.asarray(values, dtype=np.float32), window_size=window_size)
    reconstructed = model(tf.convert_to_tensor(sequences), training=False).numpy()
    return np.mean(np.square(sequences - reconstructed), axis=1)


def run_injections(
    model: keras.Model,
    test_values: np.ndarray,
    specs: list[InjectionSpec],
    window_size: int,
    smoothing_window: int,
    threshold: float,
) -> list[dict]:
    """Inject each anomaly independently into a clean copy of the test data and score attribution.

    Attribution vector = per-feature reconstruction error averaged over every
    window that contains at least one injected timestep. ``detected`` is True if
    the smoothed AE score crosses the channel threshold inside the injected span
    (same rule as the main evaluation).
    """
    records: list[dict] = []
    n_points, n_features = test_values.shape
    for spec in specs:
        injected = inject_anomaly(test_values, spec)
        # Enough left context that smoothed scores inside the span match a full-series run.
        local_start = max(0, spec.start - (window_size - 1) - (smoothing_window - 1))
        local_end = min(n_points, spec.end + window_size - 1)
        errors = window_feature_errors(model, injected[local_start:local_end], window_size)
        window_ends = local_start + np.arange(len(errors)) + window_size - 1

        covering = (window_ends >= spec.start) & (window_ends <= spec.end + window_size - 2)
        feature_scores = errors[covering].mean(axis=0)
        rank = attribution_rank(feature_scores, spec.column)

        smoothed = moving_average(errors.mean(axis=1), window=smoothing_window)
        in_span = (window_ends >= spec.start) & (window_ends < spec.end)
        detected = bool(np.any(smoothed[in_span] > threshold))

        records.append(
            {
                **spec.to_dict(),
                "n_features": int(n_features),
                "is_telemetry_column": spec.column == 0,
                "predicted_column": int(np.argmax(feature_scores)),
                "rank": rank,
                "top1_hit": rank <= 1,
                "top3_hit": rank <= 3,
                "detected": detected,
            }
        )
    return records


def finalize_detector(
    labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (test_scores > threshold).astype(int)
    return {
        **evaluate_series(labels, predictions, test_scores),
        "threshold": float(threshold),
        "predictions": predictions,
        "scores": test_scores,
    }


def evaluate_autoencoder(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    seed: int,
    injections: list[InjectionSpec] | None = None,
) -> dict:
    validation_split = 0.1 if len(train_values) >= 200 else 0.0
    model, train_sequences = fit_autoencoder(
        train_values=train_values,
        window_size=window_size,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    if len(train_sequences):
        train_scores = reconstruction_scores(model, train_sequences)
    else:
        train_scores = sequence_scores_stream(model, train_values, window_size, batch_size)
    train_scores = align_sequence_scores(train_scores, len(train_values), window_size)
    train_scores = moving_average(train_scores, window=smoothing_window)
    threshold = calculate_threshold(
        train_scores,
        method=threshold_method,
        percentile=percentile,
        k=k,
    )

    test_scores = sequence_scores_stream(model, test_values, window_size, batch_size)
    test_scores = align_sequence_scores(test_scores, len(test_values), window_size)
    test_scores = moving_average(test_scores, window=smoothing_window)
    result = finalize_detector(labels, test_scores, threshold)

    result["attribution_records"] = run_injections(
        model,
        test_values,
        injections or [],
        window_size=window_size,
        smoothing_window=smoothing_window,
        threshold=threshold,
    )
    keras.backend.clear_session()
    return result


def evaluate_isolation_forest(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    seed: int,
) -> dict:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_values)

    train_scores = moving_average(-model.score_samples(train_values), window=smoothing_window)
    threshold = calculate_threshold(train_scores, method=threshold_method, percentile=percentile, k=k)
    test_scores = moving_average(-model.score_samples(test_values), window=smoothing_window)
    return finalize_detector(labels, test_scores, threshold)


def evaluate_first_difference(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    columns: list[int] | None,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
) -> dict:
    train_scores = moving_average(first_difference_scores(train_values, columns), window=smoothing_window)
    threshold = calculate_threshold(train_scores, method=threshold_method, percentile=percentile, k=k)
    test_scores = moving_average(first_difference_scores(test_values, columns), window=smoothing_window)
    return finalize_detector(labels, test_scores, threshold)


def minmax_scale(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_min = train_values.min(axis=0)
    train_max = train_values.max(axis=0)
    denom = np.where(train_max - train_min == 0, 1.0, train_max - train_min)
    train_scaled = (train_values - train_min) / denom
    test_scaled = (test_values - train_min) / denom
    return train_scaled.astype("float32"), test_scaled.astype("float32")


def summarize_detector(results: list[dict], labels: list[np.ndarray]) -> dict:
    """Pooled metrics at the top level (backwards compatible keys) plus ``channel_mean``.

    Note: pooled ROC-AUC concatenates raw scores from independently scaled
    per-channel detectors, so treat it as indicative only; the channel-mean
    ROC-AUC is the cleaner number.
    """
    aggregated = aggregate_series_results(results)
    pooled = dict(aggregated["pooled"])
    pooled["roc_auc"] = safe_roc_auc(
        np.concatenate(labels),
        np.concatenate([r["scores"] for r in results]),
    )
    return {**pooled, "channel_mean": aggregated["channel_mean"]}


def load_aggregate_dataset(dataset_root: str | Path, spacecraft: str) -> list[dict]:
    dataset_root = Path(dataset_root)
    train_path = dataset_root / f"{spacecraft}_train.npy"
    test_path = dataset_root / f"{spacecraft}_test.npy"
    label_path = dataset_root / f"{spacecraft}_test_label.npy"
    if not train_path.exists() or not test_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Aggregate {spacecraft} dataset files were not found in {dataset_root}.")

    return [
        {
            "channel_id": spacecraft,
            "spacecraft": spacecraft,
            "class": "aggregate",
            "train": load_numpy_array(train_path),
            "test": load_numpy_array(test_path),
            "labels": np.load(label_path).astype(int).reshape(-1),
        }
    ]


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown_report(summary: dict) -> str:
    lines = [
        f"# {summary['dataset']} evaluation ({summary['channel_count']} channels)",
        "",
        f"Window {summary['window_size']}, epochs {summary['epochs']}, latent {summary['latent_dim']}, "
        f"threshold {summary['threshold_method']} (p={summary['percentile']}, k={summary['k']}) on smoothed "
        f"training scores, smoothing {summary['smoothing_window']}. Diff baseline columns: "
        f"{summary['diff_columns']}.",
        "",
        "## Detection",
        "",
        "| Model | Strict F1 (ch-mean) | Strict F1 (pooled) | Adj F1 (ch-mean) | Adj F1 (pooled) "
        "| Pooled P / R | ROC-AUC (ch-mean) | Events detected | Mean latency (steps) | False alarms / 1k pts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for prefix, key in MODEL_KEYS.items():
        s = summary[key]
        cm = s["channel_mean"]
        lines.append(
            f"| {MODEL_LABELS[prefix]} | {_fmt(cm['f1'])} | {_fmt(s['f1'])} | {_fmt(cm['adjusted_f1'])} "
            f"| {_fmt(s['adjusted_f1'])} | {_fmt(s['precision'])} / {_fmt(s['recall'])} | {_fmt(cm['roc_auc'])} "
            f"| {s['detected_events']}/{s['events']} | {_fmt(s['mean_latency'], 1)} "
            f"| {_fmt(s['false_alarms_per_1k'], 2)} |"
        )
    lines += [
        "",
        "Latency is pooled over detected events only (an alert must fall inside the labelled event). "
        "A false alarm is a contiguous alert run that does not overlap any labelled event.",
    ]

    attribution = summary.get("attribution")
    if attribution and attribution.get("n_injections"):
        lines += [
            "",
            f"## Root-cause attribution ({attribution['n_injections']} injected anomalies)",
            "",
            "| Type | n | Top-1 hit | Top-3 hit | Top-1 chance | Top-3 chance | Mean rank | AE detected |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for kind, block in attribution["by_type"].items():
            if not block.get("n"):
                continue
            lines.append(
                f"| {kind} | {block['n']} | {_fmt(block['top1_hit_rate'])} | {_fmt(block['top3_hit_rate'])} "
                f"| {_fmt(block['top1_chance'])} | {_fmt(block['top3_chance'])} | {_fmt(block['mean_rank'], 1)} "
                f"| {_fmt(block.get('detection_rate', float('nan')))} |"
            )
        split = attribution.get("by_target_column_type", {})
        if split:
            lines += [
                "",
                "| Injected into | n | Top-1 hit | Top-3 hit |",
                "|---|---|---|---|",
            ]
            for name, block in split.items():
                if block.get("n"):
                    lines.append(
                        f"| {name} | {block['n']} | {_fmt(block['top1_hit_rate'])} | {_fmt(block['top3_hit_rate'])} |"
                    )
        lines += [
            "",
            "Chance = expected hit rate of a uniformly random feature ranking.",
        ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    dataset_root: str,
    spacecraft: str,
    output_dir: str,
    window_size: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    limit_channels: int | None,
    seed: int,
    n_injections: int = 200,
    diff_columns: str = "telemetry",
) -> dict:
    try:
        channels = iter_telemanom_channels(dataset_root, spacecraft=spacecraft, limit=limit_channels)
    except FileNotFoundError:
        channels = load_aggregate_dataset(dataset_root, spacecraft=spacecraft)
        if limit_channels is not None:
            channels = channels[:limit_channels]
    if not channels:
        raise ValueError(f"No channels found for spacecraft={spacecraft} in {dataset_root}.")

    # Scale and label every channel up front so injections can be sampled across channels.
    for channel in channels:
        channel["train_scaled"], channel["test_scaled"] = minmax_scale(channel["train"], channel["test"])
        if "labels" not in channel:
            channel["labels"] = build_point_labels(len(channel["test"]), channel["anomaly_sequences"])

    diff_cols = [0] if diff_columns == "telemetry" else None
    injection_specs: list[InjectionSpec] = []
    if n_injections > 0:
        injection_specs = sample_injections(
            [
                {
                    # Key by position: the upstream metadata lists some channel ids twice (SMAP P-2).
                    "channel_id": str(position),
                    "n_points": len(c["test_scaled"]),
                    "train_std": c["train_scaled"].std(axis=0),
                    "segments": anomaly_segments(c["labels"]),
                }
                for position, c in enumerate(channels)
            ],
            n_injections=n_injections,
            rng=np.random.default_rng(seed),
            min_start=window_size + smoothing_window,
            avoid_margin=window_size,
        )
    specs_by_channel: dict[int, list[InjectionSpec]] = defaultdict(list)
    for spec in injection_specs:
        position = int(spec.channel_id)
        specs_by_channel[position].append(replace(spec, channel_id=channels[position]["channel_id"]))

    rows: list[dict] = []
    results: dict[str, list[dict]] = {prefix: [] for prefix in MODEL_KEYS}
    all_labels: list[np.ndarray] = []
    attribution_records: list[dict] = []

    started = time.time()
    for index, channel in enumerate(channels, start=1):
        channel_id = channel["channel_id"]
        train_values, test_values, labels = channel["train_scaled"], channel["test_scaled"], channel["labels"]

        per_model = {
            "ae": evaluate_autoencoder(
                train_values=train_values,
                test_values=test_values,
                labels=labels,
                window_size=window_size,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                threshold_method=threshold_method,
                percentile=percentile,
                k=k,
                smoothing_window=smoothing_window,
                seed=seed + index,
                injections=specs_by_channel.get(index - 1, []),
            ),
            "if": evaluate_isolation_forest(
                train_values=train_values,
                test_values=test_values,
                labels=labels,
                threshold_method=threshold_method,
                percentile=percentile,
                k=k,
                smoothing_window=smoothing_window,
                seed=seed + index,
            ),
            "diff": evaluate_first_difference(
                train_values=train_values,
                test_values=test_values,
                labels=labels,
                columns=diff_cols,
                threshold_method=threshold_method,
                percentile=percentile,
                k=k,
                smoothing_window=smoothing_window,
            ),
        }

        row = {
            "channel_id": channel_id,
            "class": channel.get("class", ""),
            "test_points": int(len(test_values)),
            "anomaly_points": int(labels.sum()),
        }
        for prefix, result in per_model.items():
            for key in ROW_METRIC_KEYS:
                row[f"{prefix}_{key}"] = result[key]
            results[prefix].append(result)
        rows.append(row)
        all_labels.append(labels)
        attribution_records.extend(per_model["ae"]["attribution_records"])

        ae, iso, diff = per_model["ae"], per_model["if"], per_model["diff"]
        print(
            f"[{index:02d}/{len(channels)}] {channel_id} | "
            f"AE F1={ae['f1']:.3f}/adj {ae['adjusted_f1']:.3f} | "
            f"IF F1={iso['f1']:.3f}/adj {iso['adjusted_f1']:.3f} | "
            f"Diff F1={diff['f1']:.3f}/adj {diff['adjusted_f1']:.3f} | "
            f"injections={len(ae['attribution_records'])}",
            flush=True,
        )

    summaries = {prefix: summarize_detector(results[prefix], all_labels) for prefix in MODEL_KEYS}

    attribution_summary = summarize_attribution(attribution_records)
    by_column_type = {
        "telemetry column (0)": summarize_attribution(
            [r for r in attribution_records if r["is_telemetry_column"]]
        )["overall"],
        "command columns (1+)": summarize_attribution(
            [r for r in attribution_records if not r["is_telemetry_column"]]
        )["overall"],
    }

    output_dir_path = Path(output_dir)
    ensure_parent(output_dir_path / "placeholder.txt")
    stem = spacecraft.lower()
    metrics_path = output_dir_path / f"{stem}_channel_metrics.csv"
    summary_path = output_dir_path / f"{stem}_benchmark_summary.json"
    attribution_path = output_dir_path / f"{stem}_attribution_injections.csv"
    report_path = output_dir_path / f"{stem}_evaluation_report.md"

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(attribution_records).to_csv(attribution_path, index=False)

    ae_summary, iso_summary = summaries["ae"], summaries["if"]
    summary = {
        "dataset": spacecraft,
        "channel_count": len(channels),
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "threshold_method": threshold_method,
        "percentile": percentile,
        "k": k,
        "smoothing_window": smoothing_window,
        "diff_columns": diff_columns,
        "seed": seed,
        "runtime_seconds": round(time.time() - started, 2),
        **{MODEL_KEYS[prefix]: summaries[prefix] for prefix in MODEL_KEYS},
        "attribution": {
            "n_injections": len(attribution_records),
            "method": "argmax of per-feature reconstruction MSE, averaged over windows containing the injection",
            "by_type": attribution_summary,
            "by_target_column_type": by_column_type,
        },
        "f1_improvement_vs_isolation_forest_pct": round((ae_summary["f1"] - iso_summary["f1"]) * 100, 2),
        "adjusted_f1_improvement_vs_isolation_forest_pct": round(
            (ae_summary["adjusted_f1"] - iso_summary["adjusted_f1"]) * 100,
            2,
        ),
    }
    save_json(summary, summary_path)
    report = build_markdown_report(summary)
    report_path.write_text(report, encoding="utf-8")
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "attribution_path": str(attribution_path),
        "report_path": str(report_path),
        "summary": summary,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the LSTM autoencoder on NASA SMAP/MSL.")
    parser.add_argument("--dataset-root", default="data/nasa_telemanom")
    parser.add_argument("--spacecraft", choices=["SMAP", "MSL"], default="SMAP")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--threshold-method", choices=["percentile", "std"], default="percentile")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--limit-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-injections",
        type=int,
        default=200,
        help="Synthetic anomalies injected for attribution validation (0 disables).",
    )
    parser.add_argument(
        "--diff-columns",
        choices=["telemetry", "all"],
        default="telemetry",
        help="Columns used by the |x_t - x_{t-1}| baseline: the telemetry value (column 0) or max over all.",
    )
    args = parser.parse_args()

    # Reproducibility: seed Python/NumPy/TF globally and force deterministic TF kernels
    # (per-channel models are additionally re-seeded with seed + channel index).
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    result = run_benchmark(
        dataset_root=args.dataset_root,
        spacecraft=args.spacecraft,
        output_dir=args.output_dir,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        k=args.k,
        smoothing_window=args.smoothing_window,
        limit_channels=args.limit_channels,
        seed=args.seed,
        n_injections=args.n_injections,
        diff_columns=args.diff_columns,
    )

    print(f"Saved per-channel metrics to {result['metrics_path']}")
    print(f"Saved summary to {result['summary_path']}")
    print(f"Saved attribution injections to {result['attribution_path']}")
    print(f"Saved report to {result['report_path']}")
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
