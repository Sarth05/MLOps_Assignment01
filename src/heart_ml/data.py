"""Download and normalize the processed Cleveland Heart Disease dataset."""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw UCI columns; retain missing feature values for pipeline imputation."""
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, got {frame.shape[1]}")
    frame = frame.copy()
    frame.columns = COLUMNS
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column].replace("?", pd.NA), errors="coerce")
    frame = frame.dropna(subset=["target"])
    frame["target"] = (frame["target"].astype(int) > 0).astype(int)
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary after normalization")
    return frame.reset_index(drop=True)


def download_dataset(output: str | Path) -> Path:
    """Download the UCI archive and save the processed Cleveland subset as CSV."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(UCI_ARCHIVE_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        matches = [name for name in archive.namelist()
                   if name.lower().endswith("processed.cleveland.data")]
        if not matches:
            raise FileNotFoundError("processed.cleveland.data not found in UCI archive")
        with archive.open(matches[0]) as handle:
            raw = pd.read_csv(handle, header=None, names=COLUMNS, dtype=str)
    clean = normalize_frame(raw)
    clean.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/heart.csv")
    args = parser.parse_args()
    path = download_dataset(args.output)
    frame = pd.read_csv(path)
    print(f"Saved {len(frame)} rows and {len(frame.columns)} columns to {path}")
    print(frame['target'].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
