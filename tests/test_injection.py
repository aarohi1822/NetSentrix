import numpy as np
import pytest

from injection import (
    ANOMALY_TYPES,
    SIGMA_FLOOR,
    InjectionSpec,
    attribution_rank,
    inject_anomaly,
    injection_labels,
    sample_injections,
    summarize_attribution,
    top_k_hit,
)


def spec(kind="spike", column=1, start=5, length=1, magnitude=2.0):
    return InjectionSpec("C-1", kind, column, start, length, magnitude)


class TestInject:
    def test_spike(self):
        values = np.zeros((10, 3), dtype=np.float32)
        out = inject_anomaly(values, spec())
        expected = np.zeros((10, 3))
        expected[5, 1] = 2.0
        np.testing.assert_allclose(out, expected)
        assert out.dtype == np.float32
        assert not values.any(), "input must not be mutated"

    def test_level_shift(self):
        out = inject_anomaly(np.zeros((10, 2)), spec("level_shift", column=0, start=2, length=4, magnitude=-1.5))
        np.testing.assert_allclose(out[:, 0], [0, 0, -1.5, -1.5, -1.5, -1.5, 0, 0, 0, 0])
        assert not out[:, 1].any()

    def test_drift_is_linear_ramp(self):
        out = inject_anomaly(np.ones((8, 1)), spec("drift", column=0, start=2, length=4, magnitude=4.0))
        np.testing.assert_allclose(out[:, 0], [1, 1, 2, 3, 4, 5, 1, 1])

    @pytest.mark.parametrize(
        "bad",
        [
            spec(kind="sine"),
            spec(length=0),
            spec(start=-1),
            spec(start=9, length=2),
            spec(column=3),
        ],
    )
    def test_invalid_specs(self, bad):
        with pytest.raises(ValueError):
            inject_anomaly(np.zeros((10, 3)), bad)

    def test_labels(self):
        np.testing.assert_array_equal(injection_labels(6, spec(start=2, length=3)), [0, 0, 1, 1, 1, 0])


class TestRank:
    def test_rank_and_hits(self):
        scores = [0.1, 0.9, 0.5, 0.3]
        assert attribution_rank(scores, 1) == 1
        assert attribution_rank(scores, 2) == 2
        assert attribution_rank(scores, 0) == 4
        assert top_k_hit(scores, 3, k=3)
        assert not top_k_hit(scores, 0, k=3)

    def test_ties_are_pessimistic(self):
        assert attribution_rank([1.0, 1.0, 1.0], 0) == 3
        assert attribution_rank([0.5, 1.0, 1.0], 2) == 2

    def test_nan_target_is_last(self):
        assert attribution_rank([0.1, np.nan, 0.3], 1) == 3

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            attribution_rank([0.1, 0.2], 2)


def make_channels():
    return [
        {"channel_id": "A", "n_points": 2000, "train_std": [0.2, 0.0, 0.01], "segments": [(500, 900)]},
        {"channel_id": "B", "n_points": 1500, "train_std": [0.0, 0.3, 0.0]},
        {"channel_id": "C", "n_points": 1500, "train_std": [0.0, 0.0, 0.0]},  # ineligible
    ]


class TestSample:
    def test_count_balance_and_reproducibility(self):
        specs = sample_injections(make_channels(), 200, np.random.default_rng(7), min_start=70, avoid_margin=60)
        assert len(specs) == 200
        counts = {kind: sum(s.kind == kind for s in specs) for kind in ANOMALY_TYPES}
        assert counts == {"spike": 67, "level_shift": 67, "drift": 66}
        again = sample_injections(make_channels(), 200, np.random.default_rng(7), min_start=70, avoid_margin=60)
        assert specs == again

    def test_constraints(self):
        channels = {c["channel_id"]: c for c in make_channels()}
        specs = sample_injections(make_channels(), 300, np.random.default_rng(1), min_start=70, avoid_margin=60)
        for s in specs:
            channel = channels[s.channel_id]
            std = np.asarray(channel["train_std"])
            assert s.channel_id != "C"
            assert std[s.column] > 0, "only varying columns are targeted"
            assert s.start >= 70 and s.end <= channel["n_points"]
            sigma = max(std[s.column], SIGMA_FLOOR)
            if s.kind == "spike":
                assert s.length == 1
                assert 6 * sigma - 1e-9 <= abs(s.magnitude) <= 10 * sigma + 1e-9
            elif s.kind == "level_shift":
                assert 30 <= s.length <= 150
            else:
                assert 100 <= s.length <= 300
            for seg_start, seg_end in channel.get("segments", []):
                assert s.end + 60 <= seg_start or s.start >= seg_end + 60

    def test_no_eligible_channel(self):
        with pytest.raises(ValueError):
            sample_injections([make_channels()[2]], 5, np.random.default_rng(0), min_start=10)


class TestSummarize:
    def test_hit_rates_and_chance(self):
        records = [
            {"kind": "spike", "rank": 1, "n_features": 10, "detected": True},
            {"kind": "spike", "rank": 4, "n_features": 10, "detected": False},
            {"kind": "drift", "rank": 2, "n_features": 10, "detected": True},
        ]
        s = summarize_attribution(records)
        assert s["spike"]["n"] == 2
        assert s["spike"]["top1_hit_rate"] == pytest.approx(0.5)
        assert s["spike"]["top3_hit_rate"] == pytest.approx(0.5)
        assert s["spike"]["top1_chance"] == pytest.approx(0.1)
        assert s["spike"]["top3_chance"] == pytest.approx(0.3)
        assert s["spike"]["detection_rate"] == pytest.approx(0.5)
        assert s["drift"]["top1_hit_rate"] == 0.0
        assert s["drift"]["top3_hit_rate"] == 1.0
        assert s["level_shift"] == {"n": 0}
        assert s["overall"]["n"] == 3
        assert s["overall"]["top3_hit_rate"] == pytest.approx(2 / 3)
        assert s["overall"]["mean_rank"] == pytest.approx(7 / 3)

    def test_chance_capped_for_few_features(self):
        s = summarize_attribution([{"kind": "spike", "rank": 1, "n_features": 2}])
        assert s["spike"]["top3_chance"] == pytest.approx(1.0)
        assert "detection_rate" not in s["spike"]
