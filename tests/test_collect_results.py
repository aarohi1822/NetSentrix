import json
import math

import pytest

from collect_results import RESULTS_END, RESULTS_START, _clean, render_results_markdown, update_readme


def detector(f1, adj, events=4, detected=2):
    block = {
        "precision": 0.5,
        "recall": 0.25,
        "f1": f1,
        "adjusted_f1": adj,
        "roc_auc": 0.6,
        "events": events,
        "detected_events": detected,
        "event_recall": detected / events,
        "mean_latency": None,
        "false_alarms_per_1k": 1.25,
    }
    return {"pooled": dict(block), "channel_mean": dict(block)}


def fake_results():
    attribution = {
        "by_type": {
            "spike": {"n": 2, "top1_hit_rate": 0.5, "top3_hit_rate": 1.0},
            "level_shift": {"n": 0},
            "drift": {"n": 1, "top1_hit_rate": 0.0, "top3_hit_rate": 1.0},
            "overall": {"n": 3, "top1_hit_rate": 1 / 3, "top3_hit_rate": 1.0, "top1_chance": 0.04, "top3_chance": 0.12},
        }
    }
    return {
        "source_dir": "data/processed/evaluation",
        "datasets": {
            "SMAP": {
                "channel_count": 2,
                "command": "python src/benchmark_smap.py --spacecraft SMAP",
                "detectors": {
                    "lstm_autoencoder": detector(0.1234, 0.5),
                    "isolation_forest": detector(0.2, 0.6),
                    "first_difference": detector(0.3, 0.7),
                },
                "attribution": attribution,
            }
        },
    }


def test_clean_replaces_non_finite():
    cleaned = _clean({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2}})
    assert cleaned == {"a": None, "b": [1.0, None], "c": {"d": 2}}
    json.dumps(cleaned, allow_nan=False)


def test_render_uses_values_from_results():
    md = render_results_markdown(fake_results())
    assert "| SMAP (2 ch) | LSTM-AE | 0.123 / 0.123 | 0.500 / 0.500 |" in md
    assert "| 2/4 | n/a | 1.25 |" in md  # missing latency rendered as n/a, never invented
    assert "| Spike | 0.50 / 1.00 (2) |" in md
    assert "| Level shift | n/a |" in md
    assert "| Random chance | 0.04 / 0.12 |" in md
    assert "python src/benchmark_smap.py --spacecraft SMAP" in md


def test_update_readme_replaces_only_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{RESULTS_START}\nold table\n{RESULTS_END}\noutro\n", encoding="utf-8")
    update_readme(readme, "new table")
    text = readme.read_text(encoding="utf-8")
    assert text == f"intro\n{RESULTS_START}\nnew table\n{RESULTS_END}\noutro\n"
    update_readme(readme, "newer")  # idempotent re-run
    assert "new table" not in readme.read_text(encoding="utf-8")


def test_update_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(readme, "x")
