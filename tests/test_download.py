import zipfile

import numpy as np
import pytest

import download_nasa_data as dl


def make_zip(path, prefix="data/data/", channels=("A-1", "B-2"), labels=True):
    with zipfile.ZipFile(path, "w") as archive:
        for split in ("train", "test"):
            for channel in channels:
                buffer = path.parent / f"{split}-{channel}.npy"
                np.save(buffer, np.full((5, 2), len(channel), dtype=np.float32))
                archive.write(buffer, f"{prefix}{split}/{channel}.npy")
        if labels:
            archive.writestr(f"{prefix}labeled_anomalies.csv", "chan_id,spacecraft,anomaly_sequences\n")
    return path


@pytest.fixture
def no_network(monkeypatch):
    def fail(url, destination):
        raise OSError(f"network disabled for {url}")

    monkeypatch.setattr(dl, "download_file", fail)
    monkeypatch.setattr(dl.shutil, "which", lambda name: None)


def test_install_from_nested_zip_keeps_other_files(tmp_path):
    out = tmp_path / "data"
    (out / "_old_aggregate").mkdir(parents=True)
    (out / "_old_aggregate" / "keep.npy").write_bytes(b"x")
    (out / "train").mkdir()
    (out / "train" / "stale.npy").write_bytes(b"old")

    dl.install_from_zip(make_zip(tmp_path / "kaggle.zip"), out)

    assert sorted(p.name for p in (out / "train").iterdir()) == ["A-1.npy", "B-2.npy"]
    assert sorted(p.name for p in (out / "test").iterdir()) == ["A-1.npy", "B-2.npy"]
    assert (out / "labeled_anomalies.csv").exists()
    assert (out / "_old_aggregate" / "keep.npy").exists(), "--force/zip install must not wipe other data"
    assert [p.name for p in out.iterdir() if p.name.startswith("tmp")] == [], "temp dir cleaned up"


def test_install_from_flat_zip(tmp_path):
    out = tmp_path / "data"
    dl.install_from_zip(make_zip(tmp_path / "flat.zip", prefix=""), out)
    assert dl.has_channel_data(out)


def test_zip_without_channel_data_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("readme.txt", "nothing here")
    with pytest.raises(dl.DatasetUnavailable):
        dl.install_from_zip(bad, tmp_path / "data")


def test_not_a_zip_is_rejected(tmp_path):
    fake = tmp_path / "fake.zip"
    fake.write_text("not a zip")
    with pytest.raises(dl.DatasetUnavailable):
        dl.install_from_zip(fake, tmp_path / "data")


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    out = tmp_path / "data"
    dl.install_from_zip(make_zip(tmp_path / "k.zip"), out)
    manifest = tmp_path / "m.sha256"
    assert dl.write_manifest(out, manifest) == 5  # labels + 2 train + 2 test
    assert dl.verify_dataset(out, manifest) == []

    np.save(out / "test" / "A-1.npy", np.zeros((5, 2), dtype=np.float32))
    (out / "train" / "B-2.npy").unlink()
    assert sorted(dl.verify_dataset(out, manifest)) == ["different: test/A-1.npy", "missing: train/B-2.npy"]


def test_existing_data_is_reused(tmp_path, no_network):
    out = tmp_path / "data"
    dl.install_from_zip(make_zip(tmp_path / "k.zip"), out)
    assert dl.prepare_dataset(out) == "existing"


def test_all_sources_fail_raises_with_instructions(tmp_path, no_network):
    with pytest.raises(dl.DatasetUnavailable) as info:
        dl.prepare_dataset(tmp_path / "data")
    message = str(info.value)
    assert "S3 archive unavailable" in message
    assert "Kaggle CLI not installed" in message
    assert "--from-zip" in message and dl.KAGGLE_URL in message


def test_no_silent_aggregate_fallback(tmp_path, no_network, monkeypatch):
    called = []
    monkeypatch.setattr(dl, "download_aggregate_smap", lambda out: called.append(out))
    with pytest.raises(dl.DatasetUnavailable):
        dl.prepare_dataset(tmp_path / "data")
    assert called == []
    assert dl.prepare_dataset(tmp_path / "data", allow_aggregate=True) == "aggregate"
    assert len(called) == 1


def test_from_zip_takes_priority(tmp_path, no_network):
    out = tmp_path / "data"
    assert dl.prepare_dataset(out, from_zip=make_zip(tmp_path / "k.zip")) == "zip"
    assert dl.has_channel_data(out)


def test_kaggle_missing_credentials_gives_clear_hint(tmp_path, monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "Traceback (most recent call last):\nKeyError: 'username'"

    monkeypatch.setattr(dl.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: Result())
    (tmp_path / "data").mkdir()
    with pytest.raises(dl.DatasetUnavailable, match=r"credentials missing or invalid"):
        dl.fetch_from_kaggle(tmp_path / "data")
