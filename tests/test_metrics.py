import math

import numpy as np
import pytest

from metrics import (
    adjusted_prf,
    aggregate_series_results,
    anomaly_segments,
    binary_runs,
    confusion_counts,
    count_false_alarms,
    detection_latencies,
    evaluate_series,
    false_alarms_per_1k,
    mean_detection_latency,
    point_adjust,
    prf_from_counts,
    safe_roc_auc,
    strict_prf,
)

# Two events: [2, 5) and [8, 10). One alert inside event 1 (t=3), event 2 missed,
# one false-alarm run [6, 7).
LABELS = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
PREDS = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])


class TestRuns:
    def test_binary_runs_basic(self):
        assert binary_runs(LABELS) == [(2, 5), (8, 10)]

    def test_runs_touching_edges(self):
        assert binary_runs([1, 1, 0, 1]) == [(0, 2), (3, 4)]

    def test_empty_and_all_zero(self):
        assert binary_runs([]) == []
        assert binary_runs([0, 0, 0]) == []

    def test_all_ones(self):
        assert anomaly_segments([1, 1, 1]) == [(0, 3)]

    def test_accepts_bool_and_nonbinary(self):
        assert binary_runs(np.array([False, True, True])) == [(1, 3)]
        assert binary_runs([0, 2, 5, 0]) == [(1, 3)]


class TestPointAdjust:
    def test_expands_detected_event_only(self):
        adjusted = point_adjust(LABELS, PREDS)
        expected = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(adjusted, expected)

    def test_does_not_mutate_inputs(self):
        preds = PREDS.copy()
        point_adjust(LABELS, preds)
        np.testing.assert_array_equal(preds, PREDS)

    def test_event_at_end_of_series(self):
        np.testing.assert_array_equal(point_adjust([0, 1, 1, 1], [0, 0, 0, 1]), [0, 1, 1, 1])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            point_adjust([0, 1], [0, 1, 0])


class TestPRF:
    def test_confusion_counts(self):
        assert confusion_counts(LABELS, PREDS) == {"tp": 1, "fp": 1, "fn": 4, "tn": 6}

    def test_strict(self):
        m = strict_prf(LABELS, PREDS)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(1 / 5)
        assert m["f1"] == pytest.approx(2 * 0.5 * 0.2 / 0.7)

    def test_adjusted(self):
        # adjusted: tp=3 (whole first event), fp=1, fn=2
        m = adjusted_prf(LABELS, PREDS)
        assert m["precision"] == pytest.approx(3 / 4)
        assert m["recall"] == pytest.approx(3 / 5)
        assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)

    def test_adjusted_never_below_strict_recall(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            labels = rng.random(200) < 0.1
            preds = rng.random(200) < 0.05
            assert adjusted_prf(labels, preds)["recall"] >= strict_prf(labels, preds)["recall"]

    def test_matches_sklearn(self):
        from sklearn.metrics import precision_recall_fscore_support

        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 500)
        preds = rng.integers(0, 2, 500)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        m = strict_prf(labels, preds)
        assert m["precision"] == pytest.approx(p)
        assert m["recall"] == pytest.approx(r)
        assert m["f1"] == pytest.approx(f)

    def test_zero_division(self):
        assert prf_from_counts(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        assert strict_prf([0, 1, 1], [0, 0, 0]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_perfect(self):
        assert strict_prf(LABELS, LABELS)["f1"] == pytest.approx(1.0)


class TestLatency:
    def test_per_event(self):
        assert detection_latencies(LABELS, PREDS) == [1, None]

    def test_alert_at_start_is_zero(self):
        assert detection_latencies([0, 1, 1, 0], [0, 1, 0, 0]) == [0]

    def test_alert_before_event_does_not_count(self):
        # alert run starts before the event and continues into it: latency 0
        assert detection_latencies([0, 0, 1, 1], [0, 1, 1, 0]) == [0]
        # alert ends before the event: missed
        assert detection_latencies([0, 0, 1, 1], [1, 0, 0, 0]) == [None]

    def test_mean_over_detected_only(self):
        labels = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        preds = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        assert detection_latencies(labels, preds) == [2, 0, None]
        assert mean_detection_latency(labels, preds) == pytest.approx(1.0)

    def test_nan_when_nothing_detected(self):
        assert math.isnan(mean_detection_latency([0, 1, 1], [0, 0, 0]))
        assert math.isnan(mean_detection_latency([0, 0, 0], [1, 1, 1]))


class TestFalseAlarms:
    def test_count(self):
        assert count_false_alarms(LABELS, PREDS) == 1

    def test_overlapping_run_is_not_false_alarm(self):
        # one long run overlapping an event + one disjoint run
        assert count_false_alarms([0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1]) == 1

    def test_runs_not_points(self):
        assert count_false_alarms([0] * 10, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]) == 2

    def test_per_1k(self):
        labels = np.zeros(2000, dtype=int)
        preds = np.zeros(2000, dtype=int)
        preds[[10, 20, 30]] = 1
        assert false_alarms_per_1k(labels, preds) == pytest.approx(1.5)

    def test_per_1k_empty(self):
        assert math.isnan(false_alarms_per_1k([], []))


class TestRocAuc:
    def test_single_class_is_nan(self):
        assert math.isnan(safe_roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_perfect_ranking(self):
        assert safe_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


class TestEvaluateAndAggregate:
    def test_evaluate_series_fields(self):
        r = evaluate_series(LABELS, PREDS, scores=PREDS.astype(float))
        assert r["f1"] == pytest.approx(strict_prf(LABELS, PREDS)["f1"])
        assert r["adjusted_f1"] == pytest.approx(adjusted_prf(LABELS, PREDS)["f1"])
        assert r["events"] == 2
        assert r["detected_events"] == 1
        assert r["event_recall"] == pytest.approx(0.5)
        assert r["mean_latency"] == pytest.approx(1.0)
        assert r["false_alarms"] == 1
        assert r["false_alarms_per_1k"] == pytest.approx(1000 / 12)
        assert r["points"] == 12
        assert r["anomaly_points"] == 5
        assert r["predicted_anomalies"] == 2

    def test_no_scores_gives_nan_auc(self):
        assert math.isnan(evaluate_series(LABELS, PREDS)["roc_auc"])

    def test_channel_mean_vs_pooled(self):
        # channel A: perfect; channel B: 1 tp, 3 fn, 0 fp
        a = evaluate_series([0, 1, 1, 0], [0, 1, 1, 0])
        b = evaluate_series([1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0])
        agg = aggregate_series_results([a, b])

        f1_b = 2 * 1.0 * 0.25 / 1.25
        assert agg["channel_mean"]["f1"] == pytest.approx((1.0 + f1_b) / 2)
        # pooled strict: tp=3, fp=0, fn=3
        assert agg["pooled"]["precision"] == pytest.approx(1.0)
        assert agg["pooled"]["recall"] == pytest.approx(0.5)
        assert agg["pooled"]["f1"] == pytest.approx(2 / 3)
        # adjusted: both events fully detected
        assert agg["pooled"]["adjusted_f1"] == pytest.approx(1.0)
        assert agg["pooled"]["events"] == 2
        assert agg["pooled"]["detected_events"] == 2
        assert agg["pooled"]["mean_latency"] == pytest.approx((0 + 3) / 2)
        assert agg["channel_mean"]["mean_latency"] == pytest.approx((0 + 3) / 2)
        assert agg["pooled"]["evaluated_points"] == 12

    def test_pooled_adjust_is_per_channel(self):
        # Channel A ends inside an event, channel B starts with one. If channels were
        # concatenated before adjusting, the alert in A would wrongly credit B's event.
        a = evaluate_series([0, 0, 1, 1], [0, 0, 0, 1])
        b = evaluate_series([1, 1, 0, 0], [0, 0, 0, 0])
        agg = aggregate_series_results([a, b])
        assert agg["pooled"]["adjusted_recall"] == pytest.approx(2 / 4)
        assert agg["pooled"]["events"] == 2
        assert agg["pooled"]["detected_events"] == 1

    def test_pooled_false_alarm_rate(self):
        a = evaluate_series([0] * 1000, [1] + [0] * 999)
        b = evaluate_series([0] * 1000, [1, 0, 1] + [0] * 997)
        agg = aggregate_series_results([a, b])
        assert agg["pooled"]["false_alarms"] == 3
        assert agg["pooled"]["false_alarms_per_1k"] == pytest.approx(1.5)
        assert agg["channel_mean"]["false_alarms_per_1k"] == pytest.approx(1.5)

    def test_channel_mean_ignores_nan_latency(self):
        a = evaluate_series([0, 1, 1], [0, 0, 1])  # latency 1
        b = evaluate_series([0, 1, 1], [0, 0, 0])  # undetected -> NaN
        assert aggregate_series_results([a, b])["channel_mean"]["mean_latency"] == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_series_results([])
