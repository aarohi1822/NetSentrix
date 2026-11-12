"""Download the NASA SMAP/MSL (Telemanom) per-channel dataset and verify it.

Sources, tried in order:

1. ``--from-zip PATH``: a zip you downloaded yourself, e.g. from
   https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
   (Kaggle login in the browser, no API key needed).
2. The original Telemanom S3 archive (currently returns Access Denied; kept in case it returns).
3. Kaggle via the ``kaggle`` CLI on PATH. This is what the upstream Telemanom README now
   uses. kaggle 1.x (the last line supporting Python 3.10, pinned in requirements-dev.txt)
   needs an API key in ``~/.kaggle/kaggle.json`` (kaggle.com -> Settings -> API).

If none works the script stops with instructions. The old aggregate SMAP mirror is
only used with ``--allow-aggregate``: it is a single concatenated series, and
benchmark results on it are not comparable to the per-channel results in the README.

Downloaded files are checked against ``data/nasa_telemanom.sha256``, the SHA-256 of
the exact files behind ``results/metrics.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TELEMANOM_DATA_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
HF_BASE_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
KAGGLE_DATASET = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
DEFAULT_MANIFEST = Path("data/nasa_telemanom.sha256")


class DatasetUnavailable(RuntimeError):
    pass


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_aggregate_smap(output_dir: Path) -> None:
    files = {
        "SMAP_train.npy": f"{HF_BASE_URL}/SMAP/SMAP_train.npy",
        "SMAP_test.npy": f"{HF_BASE_URL}/SMAP/SMAP_test.npy",
        "SMAP_test_label.npy": f"{HF_BASE_URL}/SMAP/SMAP_test_label.npy",
        "labeled_anomalies.csv": LABELS_URL,
    }
    for filename, url in files.items():
        download_file(url, output_dir / filename)


def find_channel_root(directory: Path) -> Path:
    """Locate the folder holding ``train/`` and ``test/`` with ``.npy`` files (any nesting depth)."""
    candidates = [directory, *sorted(p for p in directory.rglob("*") if p.is_dir())]
    for candidate in candidates:
        train, test = candidate / "train", candidate / "test"
        if train.is_dir() and test.is_dir() and any(train.glob("*.npy")) and any(test.glob("*.npy")):
            return candidate
    raise DatasetUnavailable(f"No train/ and test/ folders with .npy files found inside {directory}.")


def install_from_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a Telemanom zip and copy its train/ and test/ into ``output_dir``.

    Only ``train/`` and ``test/`` are replaced; other files in ``output_dir`` are kept.
    A ``labeled_anomalies.csv`` inside the zip is used if none exists yet.
    """
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise DatasetUnavailable(f"{zip_path} is not a zip file.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        root = find_channel_root(Path(tmp))
        for split in ("train", "test"):
            destination = output_dir / split
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(root / split, destination)
        labels = output_dir / "labeled_anomalies.csv"
        if not labels.exists():
            found = sorted(Path(tmp).rglob("labeled_anomalies.csv"))
            if found:
                shutil.copy2(found[0], labels)


def fetch_from_s3(output_dir: Path) -> None:
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        archive = Path(tmp) / "data.zip"
        try:
            download_file(TELEMANOM_DATA_URL, archive)
        except Exception as exc:  # network errors, 403, ...
            raise DatasetUnavailable(f"S3 archive unavailable ({exc}).") from exc
        install_from_zip(archive, output_dir)


def fetch_from_kaggle(output_dir: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise DatasetUnavailable("Kaggle CLI not installed (pip install kaggle).")
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        result = subprocess.run(
            [kaggle, "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmp],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stderr or result.stdout).strip()
            last = output.splitlines()[-1] if output else "unknown error"
            if any(hint in output for hint in ("'username'", "kaggle.json", "401", "Unauthorized")):
                last += " (Kaggle credentials missing or invalid: kaggle 1.x needs an API key in ~/.kaggle/kaggle.json)"
            raise DatasetUnavailable(f"Kaggle download failed: {last}")
        zips = sorted(Path(tmp).glob("*.zip"))
        if not zips:
            raise DatasetUnavailable("Kaggle download produced no zip file.")
        install_from_zip(zips[0], output_dir)


def ensure_labels(output_dir: Path) -> None:
    labels = output_dir / "labeled_anomalies.csv"
    if not labels.exists():
        download_file(LABELS_URL, labels)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_files(output_dir: Path) -> list[Path]:
    files = [output_dir / "labeled_anomalies.csv"]
    for split in ("train", "test"):
        files += sorted((output_dir / split).glob("*.npy"))
    return [f for f in files if f.exists()]


def write_manifest(output_dir: Path, manifest_path: Path) -> int:
    lines = [f"{sha256(f)}  {f.relative_to(output_dir).as_posix()}" for f in dataset_files(output_dir)]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_dataset(output_dir: Path, manifest_path: Path) -> list[str]:
    """Return a list of problems (empty = every manifest file present and identical)."""
    problems = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = output_dir / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        elif sha256(path) != expected:
            problems.append(f"different: {relative}")
    return problems


def has_channel_data(output_dir: Path) -> bool:
    return all(
        (output_dir / split).is_dir() and any((output_dir / split).glob("*.npy")) for split in ("train", "test")
    ) and (output_dir / "labeled_anomalies.csv").exists()


def unavailable_message(errors: list[str]) -> str:
    tried = "\n".join(f"  - {e}" for e in errors)
    return (
        "Could not obtain the per-channel NASA SMAP/MSL dataset.\n"
        f"Tried:\n{tried}\n\n"
        "Fix it with one of:\n"
        f"  1. Sign in to Kaggle in your browser, click Download on {KAGGLE_URL}\n"
        "     then run: python src/download_nasa_data.py --from-zip path/to/archive.zip\n"
        "  2. Put a Kaggle API key in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API) and rerun\n"
        "     (needs the kaggle CLI: pip install -r requirements-dev.txt).\n"
        "  3. (Not comparable to the README results) rerun with --allow-aggregate to use the\n"
        "     aggregate SMAP mirror."
    )


def prepare_dataset(
    output_dir: str | Path,
    force: bool = False,
    from_zip: str | Path | None = None,
    allow_aggregate: bool = False,
) -> str:
    """Make the per-channel dataset available in ``output_dir``; return the source used."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if from_zip is not None:
        install_from_zip(Path(from_zip), output_dir)
        ensure_labels(output_dir)
        return "zip"
    if has_channel_data(output_dir) and not force:
        return "existing"

    errors: list[str] = []
    for name, fetch in (("s3", fetch_from_s3), ("kaggle", fetch_from_kaggle)):
        try:
            fetch(output_dir)
            ensure_labels(output_dir)
            return name
        except DatasetUnavailable as exc:
            errors.append(str(exc))

    if allow_aggregate:
        print(
            "WARNING: per-channel data unavailable; downloading the AGGREGATE SMAP mirror.\n"
            "Benchmark results on it are NOT comparable to the per-channel results in the README."
        )
        download_aggregate_smap(output_dir)
        return "aggregate"
    raise DatasetUnavailable(unavailable_message(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the NASA SMAP/MSL per-channel dataset.")
    parser.add_argument("--output-dir", default="data/nasa_telemanom")
    parser.add_argument("--from-zip", default=None, help="Install from a zip downloaded manually (e.g. from Kaggle).")
    parser.add_argument("--force", action="store_true", help="Re-download even if train/ and test/ exist.")
    parser.add_argument("--allow-aggregate", action="store_true", help="Fall back to the non-comparable aggregate mirror.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--no-verify", action="store_true", help="Skip the SHA-256 check.")
    parser.add_argument("--write-manifest", action="store_true", help="(Maintainers) record hashes of the current files.")
    args = parser.parse_args()

    output_dir, manifest = Path(args.output_dir), Path(args.manifest)
    if args.write_manifest:
        count = write_manifest(output_dir, manifest)
        print(f"Wrote {count} hashes to {manifest}")
        return

    try:
        source = prepare_dataset(output_dir, force=args.force, from_zip=args.from_zip, allow_aggregate=args.allow_aggregate)
    except DatasetUnavailable as exc:
        raise SystemExit(str(exc))
    print(f"Dataset ready at {output_dir} (source: {source})")

    if source == "aggregate" or args.no_verify:
        return
    if not manifest.exists():
        print(f"No manifest at {manifest}; skipping verification.")
        return
    problems = verify_dataset(output_dir, manifest)
    if problems:
        shown = "\n".join(f"  {p}" for p in problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise SystemExit(
            f"Verification FAILED: {len(problems)} file(s) differ from {manifest}:\n{shown}{more}\n"
            "Benchmark results on this data may not match results/metrics.json."
        )
    print(f"Verified: all files match {manifest} (the data behind results/metrics.json).")


if __name__ == "__main__":
    main()
