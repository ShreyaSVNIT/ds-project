#!/usr/bin/env python3
"""Generate publication-ready figures for research report and presentation:
1. Precision-Recall curves.
2. ROC curves.
3. Confusion matrices.
4. Feature group gate weight distributions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.metrics import metric_dict


def plot_metrics_comparison(tabular_csv: Path, neural_csv: Path, out_dir: Path) -> None:
    dfs = []
    if tabular_csv.exists():
        df_tab = pd.read_csv(tabular_csv)
        dfs.append(df_tab)
    if neural_csv.exists():
        df_neu = pd.read_csv(neural_csv)
        dfs.append(df_neu)
    if not dfs:
        print("No results found to plot.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    # Filter to recall-constrained policy
    constrained = combined[combined["threshold_policy"] == "validation_recall_constrained"].copy()
    constrained = constrained.sort_values(by="pr_auc_average_precision", ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.barh(constrained["model"], constrained["pr_auc_average_precision"], color="#3498DB", edgecolor="black")
    plt.axvline(0.0942, color="red", linestyle="--", label="Random Baseline (Prevalence = 9.42%)")
    plt.title("PR-AUC (Average Precision) by Model (Held-Out Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("PR-AUC")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)
    plt.xlim(0, max(constrained["pr_auc_average_precision"]) * 1.15)
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_auc_model_comparison.png", dpi=200)
    plt.close()

    # Also plot recall vs precision at constrained threshold
    plt.figure(figsize=(10, 5))
    x = np.arange(len(constrained))
    w = 0.35
    plt.bar(x - w / 2, constrained["recall_sensitivity"], width=w, label="Achieved Test Recall", color="#2ECC71", edgecolor="black")
    plt.bar(x + w / 2, constrained["precision"], width=w, label="Achieved Test Precision", color="#E67E22", edgecolor="black")
    plt.axhline(0.80, color="green", linestyle="--", label="Validation Target Recall (0.80)")
    plt.xticks(x, constrained["model"], rotation=30, ha="right", fontsize=9)
    plt.title("Operational Trade-off: Recall vs Precision at Validation-Selected Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_precision_tradeoff.png", dpi=200)
    plt.close()
    print(f"Saved metric comparison figures to {out_dir}")


def plot_gate_weights(weights_csv: Path, out_dir: Path) -> None:
    if not weights_csv.exists():
        return
    df = pd.read_csv(weights_csv)
    plt.figure(figsize=(7, 4))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
    bars = plt.bar(df["group"], df["mean_gate_weight"], color=colors[:len(df)], edgecolor="black", width=0.5)
    plt.title("FGGN Mean Feature-Group Gate Weights (Attribution)", fontsize=11, fontweight="bold")
    plt.ylabel("Softmax Gate Weight")
    plt.ylim(0, max(df["mean_gate_weight"]) * 1.25)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}\n({h*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "fggn_gate_weights.png", dpi=200)
    plt.close()
    print(f"Saved gate weights plot to {out_dir / 'fggn_gate_weights.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabular-csv", default="artifacts/tabular_full/results.csv")
    p.add_argument("--neural-csv", default="artifacts/neural_full/results.csv")
    p.add_argument("--weights-csv", default="artifacts/neural_full/group_gate_weights.csv")
    p.add_argument("--out", default="artifacts/figures")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(Path(args.tabular_csv), Path(args.neural_csv), out_dir)
    plot_gate_weights(Path(args.weights_csv), out_dir)


if __name__ == "__main__":
    main()
