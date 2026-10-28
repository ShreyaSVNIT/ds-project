#!/usr/bin/env python3
"""Download the 22-column Heart Disease Health Indicators CSV to data/raw/."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/doguilmak/Heart-Diseaseor-Attack-Classification/"
    "main/heart_disease_health_indicators_BRFSS2015.csv"
)
KAGGLE_URL = "https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0 research-reproducibility"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        with destination.open("rb") as f:
            header = f.readline().decode("utf-8", errors="ignore")
        if "HeartDiseaseorAttack" not in header or destination.stat().st_size < 1_000_000:
            destination.unlink(missing_ok=True)
            raise RuntimeError("Downloaded file did not look like the expected CSV.")
        print(f"Saved dataset to {destination} ({destination.stat().st_size / (1024**2):.1f} MiB)")
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f"Automatic download failed: {exc}\nDownload the CSV from {KAGGLE_URL} and save it as:\n"
            f"  {destination}\nThe official source data/documentation are at "
            "https://www.cdc.gov/brfss/annual_data/annual_2015.html"
        ) from exc


if __name__ == "__main__":
    main()
