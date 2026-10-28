#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/tabular")
    p.add_argument("--model", default="extra_trees_balanced")
    p.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    p.add_argument("--background-size", type=int, default=100)
    p.add_argument("--explain-size", type=int, default=100)
    p.add_argument("--max-trees", type=int, default=30, help="Maximum trees to use in TreeExplainer for computational efficiency.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    model_path = run_dir / "models" / f"{args.model}.joblib"
    model = joblib.load(model_path)
    X, y = load_dataset(args.data)
    splits = make_splits(X, y, seed=args.seed)
    top_k = meta.get("top_k", X.shape[1])
    Xtr, Xv, Xte, features, _ = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k
    )
    # Use training rows for the background, never sample from test labels.
    background = Xtr.sample(min(args.background_size, len(Xtr)), random_state=args.seed)
    explain = Xte.sample(min(args.explain_size, len(Xte)), random_state=args.seed)

    # For large ensembles of deep trees, slice to a representative sub-ensemble to avoid extreme tree-traversal cost
    if hasattr(model, "estimators_") and len(model.estimators_) > args.max_trees:
        import copy
        explain_model = copy.copy(model)
        explain_model.estimators_ = model.estimators_[:args.max_trees]
        explain_model.n_estimators = len(explain_model.estimators_)
    else:
        explain_model = model

    explainer = shap.TreeExplainer(explain_model)
    shap_values = explainer.shap_values(explain)
    # Binary classifiers may return either an array or a list by SHAP version.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, -1]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, explain, show=False, max_display=min(20, explain.shape[1]))
    out = run_dir / f"shap_summary_{args.model}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {out}")
    print("Interpret SHAP as model attribution, not causation or a clinical explanation.")


if __name__ == "__main__":
    main()
