#!/usr/bin/env python3
"""Mathematical verification and formal proof of prevalence inconsistency in Khan et al. (2024).

This script formally verifies:
1. Under the natural BRFSS 2015 positive prevalence (9.42%), reported Recall (87.8%)
   and reported Precision (96.0%) mathematically require Accuracy = 98.51%,
   contradicting the published Accuracy of 91.7%.
2. Under natural prevalence (9.42%), reported Accuracy (91.7%) and reported Recall (87.8%)
   constrain maximum possible Precision to at most 53.63%.
3. The exact synthetic positive prevalence that mathematically reconciles 91.7% accuracy,
   96.0% precision, and 87.8% recall is ~52.34%, proving that published metrics are
   characteristic of an artificially balanced evaluation set (~50% prevalence),
   likely caused by applying ADASYN before CV partitioning.

Outputs:
- artifacts/methodological_audit/prevalence_inconsistency_proof.json
- artifacts/methodological_audit/prevalence_reconciliation_curve.csv
- artifacts/methodological_audit/prevalence_reconciliation_plot.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Ensure headless matplotlib and quick initialization
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "methodological_audit"


def compute_derived_metrics(pi: float, recall: float, precision: float) -> dict[str, float]:
    """Given positive prevalence pi, sensitivity (recall), and PPV (precision),
    analytically derives all confusion counts (per unit population) and accuracy.
    """
    tp = recall * pi
    fp = tp * (1.0 - precision) / precision
    n_neg = 1.0 - pi
    fn = pi - tp
    tn = n_neg - fp
    accuracy = tp + tn
    specificity = tn / n_neg if n_neg > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "prevalence": pi,
        "recall": recall,
        "precision": precision,
        "tp_rate": tp,
        "fp_rate": fp,
        "tn_rate": tn,
        "fn_rate": fn,
        "accuracy": accuracy,
        "specificity": specificity,
        "f1": f1,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Constants from Khan et al. (IEEE Access 2024) and BRFSS 2015
    pi_natural = 23893 / 253680  # 0.094185588...
    reported_acc = 0.9170
    reported_rec = 0.8780
    reported_prec = 0.9600
    reported_f1 = 0.9120

    # 2. Case A: Exact accuracy required by reported Recall + Precision under natural prevalence
    case_a = compute_derived_metrics(pi_natural, reported_rec, reported_prec)
    derived_acc_at_natural = case_a["accuracy"]

    # 3. Case B: Maximum precision possible with reported Accuracy + Recall under natural prevalence
    tp_b = reported_rec * pi_natural
    tn_b = reported_acc - tp_b
    fp_b = (1.0 - pi_natural) - tn_b
    derived_prec_at_natural = tp_b / (tp_b + fp_b)

    # 4. Case C: Find the exact synthetic prevalence pi* that reconciles 91.7% Acc, 96.0% Prec, 87.8% Rec
    # Accuracy = 1 - pi + pi * Recall * (2 - 1/Precision)
    # 0.917 = 1 - pi * [1 - Recall * (2 - 1/Precision)]
    multiplier = 1.0 - reported_rec * (2.0 - (1.0 / reported_prec))
    pi_reconciled = (1.0 - reported_acc) / multiplier
    case_c = compute_derived_metrics(pi_reconciled, reported_rec, reported_prec)

    # 5. Sweep across prevalence from 5% to 90%
    prevalences = np.linspace(0.05, 0.90, 171)
    curve_data = []
    for p in prevalences:
        res = compute_derived_metrics(p, reported_rec, reported_prec)
        curve_data.append({
            "prevalence": p,
            "required_accuracy": res["accuracy"],
            "tp_rate": res["tp_rate"],
            "fp_rate": res["fp_rate"],
            "specificity": res["specificity"],
        })
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(OUT / "prevalence_reconciliation_curve.csv", index=False)

    # 6. Save JSON report
    proof_summary = {
        "reference_paper": "Khan et al., IEEE Access 2024 (EnsCVDD-Net / BlCVDD-Net)",
        "dataset": "CDC BRFSS 2015 Heart Disease Health Indicators (253,680 rows)",
        "natural_prevalence": pi_natural,
        "reported_metrics": {
            "accuracy": reported_acc,
            "precision": reported_prec,
            "recall": reported_rec,
            "f1": reported_f1,
        },
        "formal_proof_results": {
            "proof_1_derived_accuracy_under_natural_prevalence": derived_acc_at_natural,
            "proof_1_discrepancy_vs_reported_accuracy": derived_acc_at_natural - reported_acc,
            "proof_2_max_possible_precision_given_reported_acc_and_recall": derived_prec_at_natural,
            "proof_2_discrepancy_vs_reported_precision": derived_prec_at_natural - reported_prec,
            "proof_3_reconciled_synthetic_prevalence": pi_reconciled,
            "proof_3_derived_accuracy_at_reconciled_prevalence": case_c["accuracy"],
        },
        "mathematical_conclusion": (
            f"At the true BRFSS prevalence of {pi_natural*100:.2f}%, 96.0% precision and 87.8% recall "
            f"mathematically require {derived_acc_at_natural*100:.2f}% accuracy. The reported 91.7% accuracy "
            f"is only mathematically consistent with 96% precision and 87.8% recall at a positive prevalence "
            f"of {pi_reconciled*100:.2f}%, precisely matching an artificially balanced (~50:50) test distribution."
        ),
    }

    with (OUT / "prevalence_inconsistency_proof.json").open("w", encoding="utf-8") as f:
        json.dump(proof_summary, f, indent=2)

    # 7. Generate publication figure
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(curve_df["prevalence"] * 100, curve_df["required_accuracy"] * 100,
             color="#0969da", lw=2.5, label="Required Accuracy given Prec=96%, Rec=87.8%")
    plt.axhline(reported_acc * 100, color="#cf222e", ls="--", lw=1.8,
                label=f"Reported Accuracy in Khan et al. ({reported_acc*100:.1f}%)")
    plt.axvline(pi_natural * 100, color="#1a7f37", ls=":", lw=1.8,
                label=f"Natural BRFSS 2015 Prevalence ({pi_natural*100:.2f}%)")
    plt.axvline(pi_reconciled * 100, color="#8250df", ls="-.", lw=1.8,
                label=f"Reconciled Synthetic Prevalence ({pi_reconciled*100:.2f}%)")

    plt.scatter([pi_natural * 100], [derived_acc_at_natural * 100], color="#0969da", s=70, zorder=5)
    plt.annotate(f"Natural: Acc must be {derived_acc_at_natural*100:.2f}%\n(Reported: 91.7%)",
                 xy=(pi_natural * 100, derived_acc_at_natural * 100),
                 xytext=(pi_natural * 100 + 4, derived_acc_at_natural * 100 - 3),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, weight="bold")

    plt.scatter([pi_reconciled * 100], [reported_acc * 100], color="#8250df", s=70, zorder=5)
    plt.annotate(f"Reported 91.7% requires\nPrevalence={pi_reconciled*100:.1f}% (~50:50)",
                 xy=(pi_reconciled * 100, reported_acc * 100),
                 xytext=(pi_reconciled * 100 + 4, reported_acc * 100 + 2),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, weight="bold")

    plt.title("Prevalence Inconsistency Proof: Resolving Metrics in Khan et al. (2024)", fontsize=12, pad=12)
    plt.xlabel("Positive Class Prevalence (%)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(OUT / "prevalence_reconciliation_plot.png")
    plt.close()

    print(f"Proof artifacts successfully generated in {OUT}")
    print(json.dumps(proof_summary, indent=2))


if __name__ == "__main__":
    main()

