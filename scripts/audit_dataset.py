#!/usr/bin/env python3
"""Dataset audit script for BRFSS 2015 cardiovascular health indicators.
Computes cryptographic hash, dimensions, schema checks, missingness, duplicates,
target prevalence, predictor summaries, and EDA plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import TARGET, load_dataset, make_splits, save_split_indices


def audit_dataset(data_path: str | Path, out_dir: str | Path, seed: int = 42) -> None:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Byte size & SHA-256
    file_bytes = data_path.read_bytes()
    byte_size = len(file_bytes)
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    # 2. Raw DataFrame load
    raw_df = pd.read_csv(data_path)
    n_rows, n_cols = raw_df.shape

    # 3. Target analysis
    if TARGET not in raw_df.columns:
        raise ValueError(f"Target column '{TARGET}' not in dataset columns: {list(raw_df.columns)}")

    target_counts = raw_df[TARGET].value_counts(dropna=False).to_dict()
    target_prevalence = float(raw_df[TARGET].mean())
    missing_counts = {col: int(raw_df[col].isna().sum()) for col in raw_df.columns}
    duplicate_rows = int(raw_df.duplicated().sum())

    # 4. Feature Summary
    feature_rows = []
    predictors = [col for col in raw_df.columns if col != TARGET]
    for col in predictors:
        s = raw_df[col]
        feature_rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
        })
    feature_summary_df = pd.DataFrame(feature_rows)
    feature_summary_df.to_csv(out_dir / "feature_summary.csv", index=False)

    # 5. Target Distribution CSV
    target_dist_df = pd.DataFrame([
        {"class": int(k) if pd.notna(k) else "NaN", "count": int(v), "fraction": float(v / n_rows)}
        for k, v in target_counts.items()
    ])
    target_dist_df.to_csv(out_dir / "target_distribution.csv", index=False)

    # 6. EDA Plot showing target distribution
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Negative (0)", "Positive (1)"], [target_counts.get(0.0, 0), target_counts.get(1.0, 0)],
                   color=["#4A90E2", "#D0021B"], width=0.5, edgecolor="black", linewidth=1)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 2000, f"{h:,}\n({h/n_rows*100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.title(f"Target Distribution: {TARGET} (N={n_rows:,})", fontsize=11, fontweight="bold")
    plt.ylabel("Sample Count")
    plt.ylim(0, max(target_counts.values()) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png", dpi=180)
    plt.close()

    # 7. Generate and save reference partition indices
    X, y = load_dataset(data_path)
    splits = make_splits(X, y, seed=seed)
    save_split_indices(splits, out_dir / "partition_indices.json")

    # 8. Comprehensive JSON metadata
    import platform
    import sklearn
    import torch
    audit_meta = {
        "dataset_path": str(data_path.resolve()),
        "sha256": sha256_hash,
        "byte_size": byte_size,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": list(raw_df.columns),
        "target_column": TARGET,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "positive_prevalence": target_prevalence,
        "duplicate_rows": duplicate_rows,
        "total_missing_cells": sum(missing_counts.values()),
        "split_policy": {
            "seed": seed,
            "train_size": len(splits.y_train),
            "train_prevalence": float(splits.y_train.mean()),
            "valid_size": len(splits.y_valid),
            "valid_prevalence": float(splits.y_valid.mean()),
            "test_size": len(splits.y_test),
            "test_prevalence": float(splits.y_test.mean()),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    with (out_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    print(f"Data audit complete:")
    print(f"  Rows: {n_rows:,}, Columns: {n_cols}")
    print(f"  Target: {TARGET} (Positives: {target_counts.get(1.0, 0):,}, {target_prevalence*100:.2f}%)")
    print(f"  Duplicates: {duplicate_rows:,}, Missing: {sum(missing_counts.values())}")
    print(f"  Artifacts saved to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BRFSS dataset for research integrity.")
    parser.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--out", default="artifacts/data_audit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit_dataset(args.data, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
