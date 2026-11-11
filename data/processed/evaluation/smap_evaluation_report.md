# SMAP evaluation (55 channels)

Window 60, epochs 5, latent 32, threshold percentile (p=99.0, k=3.0) on smoothed training scores, smoothing 5. Diff baseline columns: telemetry.

## Detection

| Model | Strict F1 (ch-mean) | Strict F1 (pooled) | Adj F1 (ch-mean) | Adj F1 (pooled) | Pooled P / R | ROC-AUC (ch-mean) | Events detected | Mean latency (steps) | False alarms / 1k pts |
|---|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.218 | 0.436 | 0.375 | 0.598 | 0.508 / 0.381 | 0.584 | 31/69 | 69.2 | 0.58 |
| Isolation Forest | 0.052 | 0.125 | 0.483 | 0.725 | 0.351 / 0.076 | 0.584 | 41/69 | 38.0 | 3.35 |
| |x_t - x_{t-1}| | 0.163 | 0.241 | 0.784 | 0.867 | 0.481 / 0.160 | 0.536 | 56/69 | 68.5 | 3.57 |

Latency is pooled over detected events only (an alert must fall inside the labelled event). A false alarm is a contiguous alert run that does not overlap any labelled event.

## Root-cause attribution (200 injected anomalies)

| Type | n | Top-1 hit | Top-3 hit | Top-1 chance | Top-3 chance | Mean rank | AE detected |
|---|---|---|---|---|---|---|---|
| spike | 67 | 0.179 | 0.522 | 0.040 | 0.120 | 5.6 | 0.060 |
| level_shift | 67 | 0.657 | 0.881 | 0.040 | 0.120 | 2.2 | 0.373 |
| drift | 66 | 0.636 | 0.848 | 0.040 | 0.120 | 2.6 | 0.530 |
| overall | 200 | 0.490 | 0.750 | 0.040 | 0.120 | 3.5 | 0.320 |

| Injected into | n | Top-1 hit | Top-3 hit |
|---|---|---|---|
| telemetry column (0) | 17 | 0.824 | 1.000 |
| command columns (1+) | 183 | 0.459 | 0.727 |

Chance = expected hit rate of a uniformly random feature ranking.
