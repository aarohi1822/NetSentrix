# MSL evaluation (27 channels)

Window 60, epochs 5, latent 32, threshold percentile (p=99.0, k=3.0) on smoothed training scores, smoothing 5. Diff baseline columns: telemetry.

## Detection

| Model | Strict F1 (ch-mean) | Strict F1 (pooled) | Adj F1 (ch-mean) | Adj F1 (pooled) | Pooled P / R | ROC-AUC (ch-mean) | Events detected | Mean latency (steps) | False alarms / 1k pts |
|---|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.162 | 0.191 | 0.338 | 0.331 | 0.145 / 0.278 | 0.648 | 19/36 | 73.2 | 1.06 |
| Isolation Forest | 0.122 | 0.142 | 0.606 | 0.583 | 0.128 / 0.159 | 0.580 | 25/36 | 52.6 | 5.45 |
| |x_t - x_{t-1}| | 0.145 | 0.102 | 0.760 | 0.620 | 0.094 / 0.112 | 0.617 | 31/36 | 51.4 | 3.40 |

Latency is pooled over detected events only (an alert must fall inside the labelled event). A false alarm is a contiguous alert run that does not overlap any labelled event.

## Root-cause attribution (200 injected anomalies)

| Type | n | Top-1 hit | Top-3 hit | Top-1 chance | Top-3 chance | Mean rank | AE detected |
|---|---|---|---|---|---|---|---|
| spike | 67 | 0.149 | 0.373 | 0.018 | 0.055 | 5.8 | 0.134 |
| level_shift | 67 | 0.522 | 0.687 | 0.018 | 0.055 | 2.8 | 0.612 |
| drift | 66 | 0.470 | 0.742 | 0.018 | 0.055 | 2.7 | 0.697 |
| overall | 200 | 0.380 | 0.600 | 0.018 | 0.055 | 3.8 | 0.480 |

| Injected into | n | Top-1 hit | Top-3 hit |
|---|---|---|---|
| telemetry column (0) | 12 | 0.833 | 0.917 |
| command columns (1+) | 188 | 0.351 | 0.580 |

Chance = expected hit rate of a uniformly random feature ranking.
