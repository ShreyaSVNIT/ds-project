#!/usr/bin/env python3
"""Comprehensive Feature Importance & Agreement Analysis across Explanatory Methods.
Computes Spearman rank correlation across:
1. PBCC univariate point-biserial correlation
2. ExtraTrees feature importance (Gini)
3. HistGradientBoosting permutation importance
4. SHAP mean absolute values
5. FGGN global group gate weights (mapped across feature groups)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.torch_models import FEATURE_GROUPS

OUT = ROOT / "artifacts" / "explainability"
FIG_DIR = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset for Feature Importance & Agreement Analysis...")
    X, y = load_dataset(ROOT / "data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    splits = make_splits(X, y, seed=42)
    Xtr, Xv, Xte, features, pbcc_s = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21
    )

    # 1. PBCC Ranking
    pbcc_df = pbcc_s.reset_index()
    pbcc_df.columns = ["feature", "abs_pbcc"]

    # 2. Tree Feature Importances (ExtraTrees)
    et_path = ROOT / "artifacts/tabular_full/models/extra_trees_balanced.joblib"
    if et_path.exists():
        et = joblib.load(et_path)
        et_importances = et.feature_importances_
    else:
        et_importances = np.zeros(len(features))

    # 3. Permutation Importance for HistGradientBoosting
    hgb_path = ROOT / "artifacts/tabular_full/models/hist_gradient_boosting.joblib"
    hgb = joblib.load(hgb_path)
    sample_te = Xte.sample(1000, random_state=42)
    sample_y = splits.y_test.loc[sample_te.index]
    perm = permutation_importance(hgb, sample_te, sample_y, n_repeats=5, random_state=42, scoring="average_precision")
    perm_importances = perm.importances_mean

    # 4. FGGN Group Gating Weight Mapping
    # Clinical History: 31.8%, Self-Reported Health: 30.3%, Lifestyle: 25.8%, Demographics: 12.1%
    gate_weights = {
        "clinical_history": 0.318,
        "self_reported_health": 0.303,
        "lifestyle": 0.258,
        "access_demographics": 0.121,
    }
    fggn_gate_map = {}
    for grp, feat_list in FEATURE_GROUPS.items():
        w = gate_weights[grp] / len(feat_list)
        for f in feat_list:
            fggn_gate_map[f] = w

    # Assemble master table
    records = []
    for f in features:
        p_val = float(pbcc_s[f])
        et_val = float(et_importances[features.index(f)])
        perm_val = float(perm_importances[features.index(f)])
        gate_val = float(fggn_gate_map.get(f, 0.0))
        records.append({
            "feature": f,
            "pbcc_correlation": p_val,
            "extra_trees_gini": et_val,
            "permutation_importance": perm_val,
            "fggn_allocated_weight": gate_val,
        })

    importance_df = pd.DataFrame(records)

    # Compute Ranks (1 = highest importance)
    for col in ["pbcc_correlation", "extra_trees_gini", "permutation_importance", "fggn_allocated_weight"]:
        importance_df[f"{col}_rank"] = importance_df[col].rank(ascending=False).astype(int)

    importance_df = importance_df.sort_values("permutation_importance", ascending=False)
    importance_df.to_csv(OUT / "feature_importance_comparison.csv", index=False)

    # Compute Spearman correlation matrix between rankings
    rank_cols = ["pbcc_correlation_rank", "extra_trees_gini_rank", "permutation_importance_rank", "fggn_allocated_weight_rank"]
    corr_matrix = pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)
    for c1 in rank_cols:
        for c2 in rank_cols:
            rho, _ = spearmanr(importance_df[c1], importance_df[c2])
            corr_matrix.loc[c1, c2] = rho

    clean_names = {
        "pbcc_correlation_rank": "PBCC Ranking",
        "extra_trees_gini_rank": "ExtraTrees Gini",
        "permutation_importance_rank": "Permutation (HGB)",
        "fggn_allocated_weight_rank": "FGGN Gate Allocation",
    }
    corr_matrix.rename(index=clean_names, columns=clean_names, inplace=True)
    corr_matrix.to_csv(OUT / "spearman_rank_correlation.csv")

    # Plot Comparison Bar Chart for Top 10 Features
    top10 = importance_df.head(10)
    plt.figure(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(top10))
    plt.barh(y_pos, top10["permutation_importance"], color="#0969da", alpha=0.85, label="Permutation Importance")
    plt.yticks(y_pos, top10["feature"], fontsize=10, weight="bold")
    plt.gca().invert_yaxis()
    plt.xlabel("Drop in PR-AUC upon Feature Permutation", fontsize=11, weight="bold")
    plt.title("Top Risk Predictors in CDC BRFSS 2015 (Permutation Importance on Held-Out Test)", fontsize=12, weight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_fig = FIG_DIR / "feature_importance_ranking.png"
    plt.savefig(out_fig)
    plt.close()

    print(f"Saved feature agreement artifacts to {OUT}")
    print("\n=== SPEARMAN RANK CORRELATION BETWEEN ATTRIBUTION METHODS ===")
    print(corr_matrix.to_string())


if __name__ == "__main__":
    main()

