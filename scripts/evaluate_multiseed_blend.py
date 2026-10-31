#!/usr/bin/env python3
"""Multi-seed evaluation of the HistGradientBoosting + FGGN probability blend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "hybrid_ensemble"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    preds = pd.read_csv(ROOT / "artifacts/robustness/test_predictions.csv.gz")

    weights = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0]
    seeds = [42, 101, 2024, 777, 9999]

    summary_rows = []
    for w in weights:
        seed_metrics = []
        for s in seeds:
            p_s = preds[preds["seed"] == s]
            hgb = p_s[p_s["model"] == "hist_gradient_boosting"].sort_values("row_idx")
            fggn = p_s[p_s["model"] == "feature_group_gated"].sort_values("row_idx")
            y_true = hgb["y_true"].values

            blend_prob = w * hgb["y_prob"].values + (1.0 - w) * fggn["y_prob"].values

            pr = average_precision_score(y_true, blend_prob)
            roc = roc_auc_score(y_true, blend_prob)
            brier = brier_score_loss(y_true, blend_prob)
            seed_metrics.append({"seed": s, "pr_auc": pr, "roc_auc": roc, "brier": brier})

        sdf = pd.DataFrame(seed_metrics)
        summary_rows.append({
            "weight_hist_gbt": w,
            "weight_fggn": round(1.0 - w, 2),
            "label": f"HGB_{int(w*100)}_FGGN_{int((1-w)*100)}",
            "pr_auc_mean": float(sdf["pr_auc"].mean()),
            "pr_auc_std": float(sdf["pr_auc"].std()),
            "roc_auc_mean": float(sdf["roc_auc"].mean()),
            "roc_auc_std": float(sdf["roc_auc"].std()),
            "brier_mean": float(sdf["brier"].mean()),
            "brier_std": float(sdf["brier"].std()),
        })

    res_df = pd.DataFrame(summary_rows)
    res_df.to_csv(OUT / "multiseed_blend_summary.csv", index=False)

    print("\n=== MULTI-SEED BLEND EVALUATION (5 SEEDS) ===")
    print(res_df[["label", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std", "brier_mean"]].to_string(index=False))

    best = res_df.loc[res_df["pr_auc_mean"].idxmax()]
    finding = {
        "finding": (
            "A fixed-weight exploratory scan across seven HistGradientBoosting/FGGN blends was evaluated "
            "on the persisted five-seed test predictions. The highest observed mean is reported descriptively; "
            "the weight was not selected on an independent validation layer and is not a confirmatory winner."
        ),
        "highest_observed_weight": {
            "label": best["label"],
            "pr_auc_mean": float(best["pr_auc_mean"]),
            "pr_auc_std": float(best["pr_auc_std"]),
            "roc_auc_mean": float(best["roc_auc_mean"]),
            "roc_auc_std": float(best["roc_auc_std"]),
        },
        "hgb_solo_pr_auc": float(res_df.loc[res_df["weight_hist_gbt"] == 1.0, "pr_auc_mean"].iloc[0]),
        "fggn_solo_pr_auc": float(res_df.loc[res_df["weight_hist_gbt"] == 0.0, "pr_auc_mean"].iloc[0]),
        "selection_note": "Do not interpret the maximum across tested weights as independent confirmation."
    }
    with (OUT / "blend_summary.json").open("w", encoding="utf-8") as f:
        json.dump(finding, f, indent=2)

    print(f"\nArtifacts saved in {OUT}")


if __name__ == "__main__":
    main()
