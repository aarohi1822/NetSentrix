import numpy as np

from data_loader import iter_telemanom_channels


def write_dataset(root, rows):
    (root / "train").mkdir()
    (root / "test").mkdir()
    lines = ["chan_id,spacecraft,anomaly_sequences,class,num_values"]
    for chan_id, spacecraft, sequences in rows:
        lines.append(f'{chan_id},{spacecraft},"{sequences}",[point],20')
        np.save(root / "train" / f"{chan_id}.npy", np.zeros((20, 2), dtype=np.float32))
        np.save(root / "test" / f"{chan_id}.npy", np.zeros((20, 2), dtype=np.float32))
    (root / "labeled_anomalies.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_duplicate_channel_rows_keep_first(tmp_path):
    write_dataset(
        tmp_path,
        [
            ("P-2", "SMAP", "[[5, 8]]"),
            ("A-1", "SMAP", "[[1, 2]]"),
            ("P-2", "SMAP", "[[4, 6]]"),
            ("M-1", "MSL", "[[3, 4]]"),
        ],
    )
    channels = iter_telemanom_channels(tmp_path, spacecraft="SMAP")
    assert [c["channel_id"] for c in channels] == ["P-2", "A-1"]
    assert channels[0]["anomaly_sequences"] == [(5, 8)]


def test_no_duplicates_unchanged(tmp_path):
    write_dataset(tmp_path, [("A-1", "SMAP", "[[1, 2]]"), ("B-1", "SMAP", "[]")])
    assert [c["channel_id"] for c in iter_telemanom_channels(tmp_path, "SMAP")] == ["A-1", "B-1"]
