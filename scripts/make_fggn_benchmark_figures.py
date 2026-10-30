#!/usr/bin/env python3
"""Generate publication-ready figures for the FGGN architectural improvement benchmark."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FIGS = ROOT / "artifacts" / "fggn_benchmark" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

summary_csv = ROOT / "artifacts" / "fggn_benchmark" / "ablation_summary.csv"
all_seeds_csv = ROOT / "artifacts" / "fggn_benchmark" / "all_seed_results.csv"

df_summary = pd.read_csv(summary_csv)
df_seeds = pd.read_csv(all_seeds_csv)
df_recall = df_seeds[df_seeds["threshold_policy"] == "validation_recall_constrained"]

# 1. PR-AUC Comparison Bar Chart with Error Bars
plt.figure(figsize=(10, 6))
# Sort ascending for horizontal bar chart
df_plot = df_summary.sort_values(by="pr_auc_mean", ascending=True).copy()

colors = []
for m in df_plot["model"]:
    if "hist_gradient_boosting" in m:
        colors.append("#E74C3C")  # Red for GBDT benchmarks
    elif m == "cross_group_interaction_fggn":
        colors.append("#2ECC71")  # Green for top model
    elif m == "fggn_baseline":
        colors.append("#95A5A6")  # Gray for baseline
    elif m == "enscvdd_net":
        colors.append("#BDC3C7")  # Light gray for paper
    else:
        colors.append("#3498DB")  # Blue for neural variants

bars = plt.barh(
    df_plot["model"],
    df_plot["pr_auc_mean"],
    xerr=df_plot["pr_auc_std"],
    color=colors,
    edgecolor="black",
    capsize=4,
    alpha=0.88,
)

# Reference lines
plt.axvline(0.0942, color="gray", linestyle=":", label="Random Guess (9.42% Prevalence)")
hgb_val = df_summary[df_summary["model"] == "hist_gradient_boosting"]["pr_auc_mean"].values[0]
plt.axvline(hgb_val, color="#E74C3C", linestyle="--", label=f"HistGBT Baseline ({hgb_val:.4f})")

plt.title("5-Seed PR-AUC (Average Precision) on Held-Out Test Data (N=38,052 per seed)", fontsize=11, fontweight="bold")
plt.xlabel("Test PR-AUC (Mean ± Std)", fontsize=10)
plt.xlim(0.33, 0.385)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5, fontweight="bold")

plt.legend(loc="lower right", fontsize=9)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_FIGS / "ablation_pr_auc_comparison.png", dpi=300)
plt.close()

# 2. Precision at Recall >= 80% (Screening constraint)
plt.figure(figsize=(10, 5))
df_prec = df_summary.sort_values(by="precision_mean", ascending=False).copy()
bars = plt.bar(
    range(len(df_prec)),
    df_prec["precision_mean"] * 100,
    yerr=df_prec["precision_std"] * 100,
    color="#34495E",
    edgecolor="black",
    capsize=4,
    alpha=0.85,
)
plt.xticks(range(len(df_prec)), df_prec["model"], rotation=35, ha="right", fontsize=8.5)
plt.ylabel("Precision at Recall >= 80% (%)", fontsize=10)
plt.title("Clinical Screening Precision (Guaranteed Recall >= 80% on Test)", fontsize=11, fontweight="bold")
plt.ylim(22.0, 26.0)
for i, bar in enumerate(bars):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_FIGS / "screening_precision_comparison.png", dpi=300)
plt.close()

print(f"Generated figures saved to {OUT_FIGS}")
