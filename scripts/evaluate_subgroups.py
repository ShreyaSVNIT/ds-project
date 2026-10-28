#!/usr/bin/env python3
"""Subgroup performance and robustness analysis across demographic variables.
Evaluates performance across Age, Sex, and Income strata using untouched test partitions
and predefined validation-selected operating thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def map_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw numerical survey categories to CDC BRFSS standard definitions."""
    df = df.copy()

    # Sex: 0.0 = Female, 1.0 = Male
    df["Sex_label"] = df["Sex"].map({0.0: "Female", 1.0: "Male"}).fillna("Unknown")

    # Age 13-category labels
    age_13_map = {
        1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39", 5.0: "40-44",
        6.0: "45-49", 7.0: "50-54", 8.0: "55-59", 9.0: "60-64", 10.0: "65-69",
        11.0: "70-74", 12.0: "75-79", 13.0: "80+"
    }
    df["Age_13_label"] = df["Age"].map(age_13_map).fillna("Unknown")

    # Clinically aggregated Age brackets
    def age_tier(val):
        if pd.isna(val):
            return "Unknown"
        if val <= 5.0:
            return "Young Adults (18-44)"
        elif val <= 9.0:
            return "Middle-Aged (45-64)"
        else:
            return "Older Adults (65+)"

    df["Age_bracket"] = df["Age"].apply(age_tier)

    # Income 8-category labels
    income_8_map = {
        1.0: "<$10k", 2.0: "$10k-<$15k", 3.0: "$15k-<$20k", 4.0: "$20k-<$25k",
        5.0: "$25k-<$35k", 6.0: "$35k-<$50k", 7.0: "$50k-<$75k", 8.0: ">=$75k"
    }
    df["Income_8_label"] = df["Income"].map(income_8_map).fillna("Unknown")

    # Aggregated Income tiers
    def income_tier(val):
        if pd.isna(val):
            return "Unknown"
        if val <= 4.0:
            return "Lower Income (<$25k)"
        elif val <= 6.0:
            return "Middle Income ($25k-<$50k)"
        else:
            return "Higher Income (>=$50k)"

    df["Income_tier"] = df["Income"].apply(income_tier)
    return df


def evaluate_slice(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute performance metrics and uncertainty intervals for a single group slice."""
    n = len(y_true)
    p = int(np.sum(y_true == 1))
    prevalence = p / n if n > 0 else 0.0

    unreliable = bool(n < 100 or p < 20)

    # Confusion counts
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

    rec_lower, rec_upper = wilson_interval(tp, tp + fn)
    prec_lower, prec_upper = wilson_interval(tp, tp + fp)
    spec_lower, spec_upper = wilson_interval(tn, tn + fp)

    # Ranking metrics (require both classes present)
    if p > 0 and (n - p) > 0:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    brier = float(brier_score_loss(y_true, y_prob)) if n > 0 else float("nan")

    return {
        "n_obs": n,
        "n_positives": p,
        "prevalence": prevalence,
        "unreliable_sample": unreliable,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "recall_ci_lower": rec_lower,
        "recall_ci_upper": rec_upper,
        "precision": precision,
        "precision_ci_lower": prec_lower,
        "precision_ci_upper": prec_upper,
        "specificity": specificity,
        "specificity_ci_lower": spec_lower,
        "specificity_ci_upper": spec_upper,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate subgroup performance across demographic strata.")
    parser.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--predictions", default="artifacts/robustness/test_predictions.csv.gz")
    parser.add_argument("--out", default="artifacts/subgroups")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw survey features from {args.data}...")
    raw_df = pd.read_csv(args.data, usecols=["Sex", "Age", "Income"])
    raw_df = map_demographics(raw_df)

    print(f"Loading test predictions from {args.predictions}...")
    preds_df = pd.read_csv(args.predictions)

    # Focus on the primary screening policy (recall-constrained)
    policy_name = "validation_recall_constrained"
    preds_df = preds_df[preds_df["threshold_policy"] == policy_name].copy()

    # Join demographic variables via row_idx
    print("Mapping test predictions to demographic columns...")
    preds_df["Sex_label"] = raw_df["Sex_label"].iloc[preds_df["row_idx"]].values
    preds_df["Age_bracket"] = raw_df["Age_bracket"].iloc[preds_df["row_idx"]].values
    preds_df["Age_13_label"] = raw_df["Age_13_label"].iloc[preds_df["row_idx"]].values
    preds_df["Income_tier"] = raw_df["Income_tier"].iloc[preds_df["row_idx"]].values
    preds_df["Income_8_label"] = raw_df["Income_8_label"].iloc[preds_df["row_idx"]].values

    dimensions = [
        ("Sex", "Sex_label"),
        ("Age_Bracket", "Age_bracket"),
        ("Age_Detailed", "Age_13_label"),
        ("Income_Tier", "Income_tier"),
        ("Income_Detailed", "Income_8_label"),
    ]

    records = []
    models = sorted(preds_df["model"].unique())
    seeds = sorted(preds_df["seed"].unique())

    print(f"Evaluating {len(models)} models across {len(seeds)} seeds...")

    for (dim_name, col_name) in dimensions:
        unique_groups = sorted(preds_df[col_name].unique())
        for grp in unique_groups:
            grp_mask = preds_df[col_name] == grp
            for model_name in models:
                for seed in seeds:
                    sub = preds_df[grp_mask & (preds_df["model"] == model_name) & (preds_df["seed"] == seed)]
                    if len(sub) == 0:
                        continue
                    y_true = sub["y_true"].to_numpy(dtype=int)
                    y_prob = sub["y_prob"].to_numpy(dtype=float)
                    y_pred = sub["y_pred"].to_numpy(dtype=int)

                    metrics = evaluate_slice(y_true, y_prob, y_pred)
                    row = {
                        "dimension": dim_name,
                        "subgroup": grp,
                        "model": model_name,
                        "seed": seed,
                        "threshold_policy": policy_name,
                    }
                    row.update(metrics)
                    records.append(row)

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_dir / "subgroup_results_by_seed.csv", index=False)
    print(f"Saved per-seed subgroup results to {out_dir / 'subgroup_results_by_seed.csv'}")

    # Aggregated summary across 5 seeds
    summary_cols = ["prevalence", "recall", "precision", "specificity", "f1", "pr_auc", "roc_auc", "brier_score"]
    agg_funcs = {c: ["mean", "std"] for c in summary_cols}
    agg_funcs["n_obs"] = "first"
    agg_funcs["n_positives"] = "first"
    agg_funcs["unreliable_sample"] = "any"

    summary_df = results_df.groupby(["dimension", "subgroup", "model"]).agg(agg_funcs)
    # Flatten multi-index
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(out_dir / "subgroup_summary_mean.csv", index=False)
    print(f"Saved mean subgroup summary to {out_dir / 'subgroup_summary_mean.csv'}")

    # Generate Publication Figure
    print("Generating publication subgroup visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    # Focus on top tabular model (HistGradientBoosting) and proposed neural model (FGGN)
    target_models = ["hist_gradient_boosting", "feature_group_gated"]
    plot_df = summary_df[summary_df["model"].isin(target_models) & summary_df["dimension"].isin(["Sex", "Age_Bracket", "Income_Tier"])].copy()

    # Panel 1: Recall (Sensitivity)
    ax1 = axes[0]
    subgroups = plot_df["subgroup"].unique()
    x = np.arange(len(subgroups))
    width = 0.35

    hgb_df = plot_df[plot_df["model"] == "hist_gradient_boosting"].set_index("subgroup").reindex(subgroups)
    fggn_df = plot_df[plot_df["model"] == "feature_group_gated"].set_index("subgroup").reindex(subgroups)

    ax1.bar(x - width/2, hgb_df["recall_mean"], width, yerr=hgb_df["recall_std"], label="HistGradientBoosting", color="#2980B9", capsize=3)
    ax1.bar(x + width/2, fggn_df["recall_mean"], width, yerr=fggn_df["recall_std"], label="FGGN (Gated Neural)", color="#E67E22", capsize=3)
    ax1.axhline(0.80, color="red", linestyle="--", alpha=0.8, label="Target Sensitivity (80%)")
    ax1.set_title("Operating Sensitivity (Recall) across Subgroups", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sensitivity (Recall)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subgroups, rotation=35, ha="right", fontsize=9)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8)

    # Panel 2: Precision (PPV) & Prevalence dynamics
    ax2 = axes[1]
    ax2.bar(x - width/2, hgb_df["precision_mean"], width, yerr=hgb_df["precision_std"], label="HistGradientBoosting", color="#2980B9", capsize=3)
    ax2.bar(x + width/2, fggn_df["precision_mean"], width, yerr=fggn_df["precision_std"], label="FGGN (Gated Neural)", color="#E67E22", capsize=3)
    # Overlay positive prevalence
    ax2.plot(x, hgb_df["prevalence_mean"], "ko--", label="Subgroup Positive Prevalence", linewidth=1.5, markersize=5)
    ax2.set_title("Precision & Disease Prevalence Dynamics", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision / Prevalence")
    ax2.set_xticks(x)
    ax2.set_xticklabels(subgroups, rotation=35, ha="right", fontsize=9)
    ax2.set_ylim(0.0, 0.6)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    # Panel 3: PR-AUC (Ranking)
    ax3 = axes[2]
    ax3.bar(x - width/2, hgb_df["pr_auc_mean"], width, yerr=hgb_df["pr_auc_std"], label="HistGradientBoosting", color="#2980B9", capsize=3)
    ax3.bar(x + width/2, fggn_df["pr_auc_mean"], width, yerr=fggn_df["pr_auc_std"], label="FGGN (Gated Neural)", color="#E67E22", capsize=3)
    ax3.set_title("PR-AUC Ranking Performance across Subgroups", fontsize=11, fontweight="bold")
    ax3.set_ylabel("PR-AUC")
    ax3.set_xticks(x)
    ax3.set_xticklabels(subgroups, rotation=35, ha="right", fontsize=9)
    ax3.set_ylim(0.0, 0.6)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8)

    plt.suptitle("Subgroup Robustness & Prevalence Sensitivity Analysis (BRFSS 2015 Untouched Test Partitions)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "subgroup_metrics_comparison.png", dpi=200)
    plt.close()
    print(f"Saved subgroup comparison figure to {out_dir / 'subgroup_metrics_comparison.png'}")


if __name__ == "__main__":
    main()
