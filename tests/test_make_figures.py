import json
import shutil
from pathlib import Path

import pytest

make_figures = pytest.importorskip("make_figures")  # needs matplotlib (requirements-dev.txt)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "processed" / "evaluation"
METRICS = ROOT / "results" / "metrics.json"


@pytest.fixture
def copies(tmp_path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    for ds in ("smap", "msl"):
        shutil.copy(EVAL_DIR / f"{ds}_benchmark_summary.json", eval_dir)
    metrics = tmp_path / "metrics.json"
    shutil.copy(METRICS, metrics)
    return eval_dir, metrics


def test_saved_files_agree_and_values_come_from_them(copies):
    eval_dir, metrics = copies
    values = make_figures.load_values(eval_dir, metrics)
    summary = json.loads((eval_dir / "smap_benchmark_summary.json").read_text())
    ae = values["SMAP"]["detectors"]["lstm_autoencoder"]
    assert ae["detected_events"] == summary["lstm_autoencoder"]["detected_events"]
    assert ae["false_alarms_per_1k"] == summary["lstm_autoencoder"]["false_alarms_per_1k"]
    spike = values["MSL"]["attribution"]["spike"]
    assert spike["top3_hit_rate"] == json.loads((eval_dir / "msl_benchmark_summary.json").read_text())[
        "attribution"]["by_type"]["spike"]["top3_hit_rate"]


def test_mismatch_between_sources_fails_and_writes_nothing(copies, tmp_path):
    eval_dir, metrics = copies
    path = eval_dir / "msl_benchmark_summary.json"
    summary = json.loads(path.read_text())
    summary["isolation_forest"]["false_alarms_per_1k"] += 0.01
    summary["attribution"]["by_type"]["drift"]["top3_hit_rate"] = 0.99
    path.write_text(json.dumps(summary))

    with pytest.raises(make_figures.SourceMismatch) as info:
        make_figures.load_values(eval_dir, metrics)
    message = str(info.value)
    assert "MSL.isolation_forest.false_alarms_per_1k" in message
    assert "MSL.attribution.drift.top3_hit_rate" in message
    assert not list(tmp_path.glob("*.png"))
