#!/usr/bin/env python3
"""Leakage-aware tabular baseline and ensemble evaluation pipeline.
Evaluates Majority Dummy, Logistic Regression, Random Forest, Extra Trees,
Histogram Gradient Boosting, and a Leakage-Safe Stacked Blender.
Preserves untouched natural prevalence on validation and test sets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.preprocessing import resample_train
from cardio.tabular_models import LeakageSafeStackedBlender, build_tabular_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware tabular evaluation on BRFSS-derived heart disease indicators.")
    parser.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--out", default="artifacts/tabular")
    parser.add_argument("--resampling", choices=["none", "adasyn", "random"], default="none")
    parser.add_argument("--top-k", type=int, default=21, help="PBCC-selected predictor count; 21 keeps all inputs.")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees/boosting iterations for a smoke test.")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data}...")
    X, y = load_dataset(args.data)
    splits = make_splits(X, y, seed=args.seed)

    print(f"Fitting PBCC feature selection on train partition only (top_k={args.top_k})...")
    Xtr, Xv, Xte, features, pbcc = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, args.top_k
    )

    print(f"Applying resampling ({args.resampling}) to single-model training partition...")
    Xfit, yfit = resample_train(Xtr, splits.y_train, args.resampling, args.seed)
    print(f"  Train size before resampling: {len(splits.y_train):,} ({splits.y_train.mean()*100:.2f}% pos)")
    print(f"  Train size after resampling:  {len(yfit):,} ({yfit.mean()*100:.2f}% pos)")
    print(f"  Validation size (untouched):   {len(splits.y_valid):,} ({splits.y_valid.mean()*100:.2f}% pos)")
    print(f"  Test size (untouched):         {len(splits.y_test):,} ({splits.y_test.mean()*100:.2f}% pos)")

    models = build_tabular_models(seed=args.seed, resampling=args.resampling, quick=args.quick)
    results = []

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()
        if isinstance(model, LeakageSafeStackedBlender):
            # Blender handles fold-isolated resampling internally on un-resampled train
            model.fit(Xtr, splits.y_train)
        else:
            model.fit(Xfit, yfit)
        fit_seconds = time.time() - start

        # Inference on untouched validation and test
        valid_prob = model.predict_proba(Xv)[:, 1]
        test_prob = model.predict_proba(Xte)[:, 1]

        # Select operating threshold exclusively on validation probabilities
        recall_threshold = choose_threshold_for_recall(splits.y_valid, valid_prob, args.target_recall)
        val_recall_at_thresh = float(np.mean(valid_prob[splits.y_valid == 1] >= recall_threshold))
        val_prec_at_thresh = float(np.sum((valid_prob >= recall_threshold) & (splits.y_valid == 1)) / max(np.sum(valid_prob >= recall_threshold), 1))

        for threshold_name, threshold in [("fixed_0.50", 0.5), ("validation_recall_constrained", recall_threshold)]:
            metrics = metric_dict(splits.y_test, test_prob, threshold)
            results.append({
                "model": name,
                "resampling": args.resampling,
                "top_k": len(features),
                "threshold_policy": threshold_name,
                "validation_threshold": round(float(threshold), 6),
                "val_recall_achieved": round(val_recall_at_thresh, 4) if threshold_name != "fixed_0.50" else None,
                "val_prec_achieved": round(val_prec_at_thresh, 4) if threshold_name != "fixed_0.50" else None,
                "fit_seconds": round(fit_seconds, 3),
                **metrics,
            })

        if name != "majority_dummy":
            joblib.dump(model, out / "models" / f"{name}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "results.csv", index=False)
    pbcc.rename("abs_pbcc").to_csv(out / "pbcc_scores.csv", header=True)

    with (out / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "resampling": args.resampling,
            "top_k": len(features),
            "selected_features": features,
            "n_rows": len(X),
            "n_predictors": X.shape[1],
            "positive_prevalence": float(y.mean()),
            "train_size_before_resampling": len(splits.y_train),
            "train_size_after_resampling": len(yfit),
            "validation_size": len(splits.y_valid),
            "test_size": len(splits.y_test),
            "test_prevalence": float(splits.y_test.mean()),
            "target_recall": args.target_recall,
            "evaluation_guarantee": "Validation and test partitions maintain untouched natural prevalence; zero synthetic leakage in OOF stacking."
        }, f, indent=2)

    print("\n=== TABULAR EXPERIMENT RESULTS (ON UNTOUCHED TEST SET) ===")
    summary_cols = ["model", "threshold_policy", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc_average_precision"]
    print(results_df[summary_cols].to_string(index=False))
    print(f"\nArtifacts successfully written to {out.resolve()}")


if __name__ == "__main__":
    main()
