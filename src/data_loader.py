from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path_or_buffer: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df


def split_features_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    labels = df[label_col] if label_col in df.columns else None
    timestamps = df[timestamp_col] if timestamp_col and timestamp_col in df.columns else None

    drop_cols = [col for col in [label_col, timestamp_col] if col and col in df.columns]
    features = df.drop(columns=drop_cols)
    features = features.select_dtypes(include="number")

    if features.empty:
        raise ValueError("No numeric sensor columns were found in the CSV.")
    return features, labels, timestamps


def normal_rows(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in df.columns:
        return df
    return df[df[label_col] == 0].copy()


def load_numpy_array(path: str | Path) -> np.ndarray:
    array = np.load(Path(path))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D telemetry array, got shape {array.shape}.")
    return array.astype("float32")


def parse_anomaly_sequences(raw_value: str) -> list[tuple[int, int]]:
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return []

    parsed = ast.literal_eval(str(raw_value))
    return [(int(start), int(end)) for start, end in parsed]


def build_point_labels(length: int, anomaly_sequences: list[tuple[int, int]]) -> np.ndarray:
    labels = np.zeros(length, dtype=int)
    for start, end in anomaly_sequences:
        start = max(0, int(start))
        end = min(length - 1, int(end))
        if end >= start:
            labels[start : end + 1] = 1
    return labels


def load_telemanom_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    metadata.columns = [col.strip().lower().replace(" ", "_") for col in metadata.columns]
    if "channel_id" in metadata.columns and "chan_id" not in metadata.columns:
        metadata = metadata.rename(columns={"channel_id": "chan_id"})
    if "chan_id" not in metadata.columns or "anomaly_sequences" not in metadata.columns:
        raise ValueError("Metadata CSV must include chan_id and anomaly_sequences columns.")

    metadata["anomaly_ranges"] = metadata["anomaly_sequences"].apply(parse_anomaly_sequences)
    return metadata


def load_telemanom_channel(dataset_root: str | Path, channel_id: str) -> tuple[np.ndarray, np.ndarray]:
    dataset_root = Path(dataset_root)
    train_path = dataset_root / "train" / f"{channel_id}.npy"
    test_path = dataset_root / "test" / f"{channel_id}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing SMAP/MSL arrays for channel {channel_id}.")
    return load_numpy_array(train_path), load_numpy_array(test_path)


def iter_telemanom_channels(
    dataset_root: str | Path,
    spacecraft: str = "SMAP",
    limit: int | None = None,
) -> list[dict]:
    dataset_root = Path(dataset_root)
    metadata = load_telemanom_metadata(dataset_root / "labeled_anomalies.csv")
    rows = metadata[metadata["spacecraft"].str.upper() == spacecraft.upper()].copy()
    # The upstream metadata lists some channels twice (SMAP P-2, with slightly different
    # anomaly ranges). Keep the first row so each channel is evaluated once.
    duplicated = rows["chan_id"].duplicated(keep="first")
    if duplicated.any():
        print(
            f"Note: dropping duplicate metadata rows for {sorted(rows.loc[duplicated, 'chan_id'].unique())} "
            "(keeping the first occurrence)."
        )
        rows = rows[~duplicated]
    if limit is not None:
        rows = rows.head(limit)

    channels: list[dict] = []
    for row in rows.to_dict(orient="records"):
        train_values, test_values = load_telemanom_channel(dataset_root, row["chan_id"])
        channels.append(
            {
                "channel_id": row["chan_id"],
                "spacecraft": row["spacecraft"],
                "anomaly_sequences": row["anomaly_ranges"],
                "num_values": int(row.get("num_values", len(test_values))),
                "class": row.get("class", ""),
                "train": train_values,
                "test": test_values,
            }
        )
    return channels
