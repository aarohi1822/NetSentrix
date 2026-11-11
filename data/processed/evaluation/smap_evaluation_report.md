# SMAP evaluation (54 channels)

Window 60, epochs 5, latent 32, threshold percentile (p=99.0, k=3.0) on smoothed training scores, smoothing 5. Diff baseline columns: telemetry.

## Detection

| Model | Strict F1 (ch-mean) | Strict F1 (pooled) | Adj F1 (ch-mean) | Adj F1 (pooled) | Pooled P / R | ROC-AUC (ch-mean) | Events detected | Mean latency (steps) | False alarms / 1k pts |
|---|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.204 | 0.424 | 0.365 | 0.590 | 0.497 / 0.370 | 0.577 | 30/68 | 70.1 | 0.58 |
| Isolation Forest | 0.051 | 0.126 | 0.473 | 0.717 | 0.342 / 0.077 | 0.578 | 40/68 | 38.2 | 3.38 |
| |x_t - x_{t-1}| | 0.154 | 0.227 | 0.782 | 0.867 | 0.469 / 0.150 | 0.531 | 55/68 | 69.7 | 3.27 |

Latency is pooled over detected events only (an alert must fall inside the labelled event). A false alarm is a contiguous alert run that does not overlap any labelled event.

## Root-cause attribution (200 injected anomalies)

| Type | n | Top-1 hit | Top-3 hit | Top-1 chance | Top-3 chance | Mean rank | AE detected |
|---|---|---|---|---|---|---|---|
| spike | 67 | 0.284 | 0.657 | 0.040 | 0.120 | 4.5 | 0.030 |
| level_shift | 67 | 0.642 | 0.866 | 0.040 | 0.120 | 2.1 | 0.343 |
| drift | 66 | 0.621 | 0.788 | 0.040 | 0.120 | 2.6 | 0.470 |
| overall | 200 | 0.515 | 0.770 | 0.040 | 0.120 | 3.1 | 0.280 |

| Injected into | n | Top-1 hit | Top-3 hit |
|---|---|---|---|
| telemetry column (0) | 11 | 0.818 | 1.000 |
| command columns (1+) | 189 | 0.497 | 0.757 |

Chance = expected hit rate of a uniformly random feature ranking.
