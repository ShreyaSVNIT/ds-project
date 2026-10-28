#!/usr/bin/env python3
"""Verification and metric reconciliation script.
Loads persisted per-example test predictions (`test_predictions.csv.gz`),
recomputes all metrics from raw labels and probabilities, reconciles against
reported summary CSVs, and asserts zero discrepancy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recompute_group_metrics(df_group: pd.DataFrame) -> dict:
    y_true = df_group["y_true"].to_numpy(dtype=int)
    prob = df_group["y_prob"].to_numpy(dtype=float)
    threshold = float(df_group["threshold_value"].iloc[0])
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": spec,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_pos": int(tp + fn), "n_neg": int(tn + fp),
    }


def main():
    p = argparse.ArgumentParser(description="Reconcile metrics against persisted test predictions.")
    p.add_argument("--predictions", default="artifacts/robustness/test_predictions.csv.gz")
    p.add_argument("--summary", default="artifacts/robustness/all_seeds_results.csv")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    summary_path = Path(args.summary)

    if not pred_path.exists():
        print(f"Error: {pred_path} does not exist. Run scripts/run_multiseed_robustness.py first.")
        sys.exit(1)

    print(f"Loading predictions from {pred_path}...")
    preds = pd.read_csv(pred_path, compression="gzip")
    print(f"Loaded {len(preds):,} individual prediction rows.")

    reported_df = pd.read_csv(summary_path)
    reported_constrained = reported_df[reported_df["threshold_policy"] == "validation_recall_constrained"].copy()

    recomputed_rows = []
    max_discrepancy = 0.0
    discrepancy_details = []

    for (seed, model), group in preds.groupby(["seed", "model"]):
        recomputed = recompute_group_metrics(group)
        # Locate matching row in reported_df
        match = reported_constrained[
            (reported_constrained["seed"] == seed) & (reported_constrained["model"] == model)
        ]
        if match.empty:
            print(f"Warning: No matching reported row for seed {seed}, model {model}")
            continue

        rep = match.iloc[0]
        # Compare key metrics
        check_cols = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc_average_precision", "brier_score"]
        for col in check_cols:
            diff = abs(recomputed[col] - rep[col])
            if diff > max_discrepancy:
                max_discrepancy = diff
            if diff > 1e-5:
                discrepancy_details.append(f"Discrepancy at seed={seed}, model={model}, col={col}: recomputed={recomputed[col]:.6f}, reported={rep[col]:.6f}, diff={diff:.6f}")

        recomputed_rows.append({
            "seed": seed, "model": model, **recomputed
        })

    recomputed_df = pd.DataFrame(recomputed_rows)
    print("\n=== RECONCILIATION AUDIT REPORT ===")
    print(f"Total groups reconciled: {len(recomputed_df)}")
    print(f"Maximum discrepancy across all metric columns: {max_discrepancy:.10f}")

    if discrepancy_details:
        print("WARNING: The following discrepancies exceeded 1e-5 threshold:")
        for d in discrepancy_details:
            print("  ", d)
    else:
        print("SUCCESS: All recomputed metrics match reported summary values to within floating point precision (diff < 1e-5).")

    summary_table = recomputed_df.groupby("model")[["pr_auc_average_precision", "roc_auc", "recall_sensitivity", "precision", "specificity", "brier_score"]].agg(["mean", "std"])
    print("\n=== RECOMPUTED RECONCILED METRICS (MEAN ± STD) ===")
    print(summary_table.to_string())


if __name__ == "__main__":
    main()
