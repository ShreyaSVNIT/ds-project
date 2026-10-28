#!/usr/bin/env python3
"""Probability calibration and statistical paired comparison script.
Evaluates:
1. Constant prevalence baseline Brier score and Log Loss.
2. Uncalibrated vs. Platt-scaled vs. Isotonic-calibrated predictions on holdout test sets.
3. Reliability diagrams for all five models across probability bins.
4. Paired PR-AUC and ROC-AUC differences across seeds (FGGN vs Uniform, FGGN vs EnsCVDD, Grouped vs HGB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.torch_models import EnsCVDDNet, FeatureGroupGatedNet, UniformGroupNet


from scipy.special import logit as scipy_logit


def compute_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Fit logistic calibration curve logit(P(Y=1)) = alpha + beta * logit(y_prob).
    Returns (intercept alpha, slope beta).
    Ideal: alpha = 0.0 (calibration-in-the-large), beta = 1.0 (spread of risk).
    """
    p_clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    log_odds = scipy_logit(p_clipped).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", penalty=None)
    lr.fit(log_odds, y_true)
    alpha = float(lr.intercept_[0])
    beta = float(lr.coef_[0][0])
    return alpha, beta


def get_torch_probs(model_class, pt_path, A_valid, A_test, features):
    if model_class == EnsCVDDNet:
        model = model_class(len(features))
    else:
        model = model_class(features)
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        pv = torch.sigmoid(model(torch.from_numpy(A_valid))).numpy()
        pt = torch.sigmoid(model(torch.from_numpy(A_test))).numpy()
    return pv, pt


def main():
    p = argparse.ArgumentParser(description="Evaluate probability calibration and paired differences.")
    p.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    p.add_argument("--models-dir", default="artifacts/robustness/models")
    p.add_argument("--out", default="artifacts/calibration")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 777, 9999])
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    print(f"Loading dataset from {args.data}...")
    X, y = load_dataset(args.data)

    models_info = [
        ("hist_gradient_boosting", "tabular"),
        ("logistic_regression", "tabular"),
        ("feature_group_gated", "torch", FeatureGroupGatedNet),
        ("uniform_group_net", "torch", UniformGroupNet),
        ("enscvdd_net", "torch", EnsCVDDNet),
    ]

    calibration_records = []
    # To store curves and stats for plotting seed 42 (exploratory)
    curve_data = {}

    for seed in args.seeds:
        print(f"\nEvaluating calibration for Seed {seed}...")
        splits = make_splits(X, y, seed=seed)
        Xtr, Xv, Xte, features, _ = select_top_pbcc(splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21)

        scaler = StandardScaler().fit(Xtr)
        A_valid = scaler.transform(Xv).astype(np.float32)
        A_test = scaler.transform(Xte).astype(np.float32)

        y_valid = splits.y_valid.to_numpy(dtype=int)
        y_test = splits.y_test.to_numpy(dtype=int)

        # Baseline: constant positive prevalence from training fold
        p_train_prev = float(splits.y_train.mean())
        base_probs_test = np.full_like(y_test, fill_value=p_train_prev, dtype=float)
        baseline_brier = float(brier_score_loss(y_test, base_probs_test))
        baseline_logloss = float(log_loss(y_test, base_probs_test))

        for entry in models_info:
            m_name = entry[0]
            m_type = entry[1]

            if m_type == "tabular":
                m_path = models_dir / f"{m_name}_seed_{seed}.joblib"
                m_obj = joblib.load(m_path)
                pv = m_obj.predict_proba(A_valid)[:, 1]
                pt = m_obj.predict_proba(A_test)[:, 1]
            else:
                m_class = entry[2]
                pt_path = models_dir / f"{m_name}_seed_{seed}.pt"
                pv, pt = get_torch_probs(m_class, pt_path, A_valid, A_test, features)

            # 1. Uncalibrated test metrics
            uncal_brier = float(brier_score_loss(y_test, pt))
            uncal_logloss = float(log_loss(y_test, pt))
            uncal_alpha, uncal_beta = compute_calibration_slope_intercept(y_test, pt)
            pr_auc = float(average_precision_score(y_test, pt))
            roc_auc = float(roc_auc_score(y_test, pt))

            # 2. Platt Scaling (Logistic Regression on validation predictions only)
            platt = LogisticRegression(solver="lbfgs")
            platt.fit(pv.reshape(-1, 1), y_valid)
            pt_platt = platt.predict_proba(pt.reshape(-1, 1))[:, 1]
            platt_brier = float(brier_score_loss(y_test, pt_platt))
            platt_logloss = float(log_loss(y_test, pt_platt))
            platt_alpha, platt_beta = compute_calibration_slope_intercept(y_test, pt_platt)

            # 3. Isotonic Regression (fitted on validation predictions only)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pv, y_valid)
            pt_iso = iso.predict(pt)
            iso_brier = float(brier_score_loss(y_test, pt_iso))
            iso_logloss = float(log_loss(y_test, pt_iso))
            iso_alpha, iso_beta = compute_calibration_slope_intercept(y_test, pt_iso)

            calibration_records.append({
                "seed": seed,
                "model": m_name,
                "baseline_prevalence": p_train_prev,
                "baseline_brier": baseline_brier,
                "baseline_logloss": baseline_logloss,
                "uncalibrated_brier": uncal_brier,
                "uncalibrated_logloss": uncal_logloss,
                "uncalibrated_intercept": uncal_alpha,
                "uncalibrated_slope": uncal_beta,
                "platt_brier": platt_brier,
                "platt_logloss": platt_logloss,
                "platt_intercept": platt_alpha,
                "platt_slope": platt_beta,
                "isotonic_brier": iso_brier,
                "isotonic_logloss": iso_logloss,
                "isotonic_intercept": iso_alpha,
                "isotonic_slope": iso_beta,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            })

            # Save curve points for seed 42
            if seed == 42:
                prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, pt, n_bins=10)
                prob_true_platt, prob_pred_platt = calibration_curve(y_test, pt_platt, n_bins=10)
                prob_true_iso, prob_pred_iso = calibration_curve(y_test, pt_iso, n_bins=10)
                curve_data[m_name] = {
                    "uncal": (prob_pred_uncal, prob_true_uncal, uncal_alpha, uncal_beta),
                    "platt": (prob_pred_platt, prob_true_platt, platt_alpha, platt_beta),
                    "iso": (prob_pred_iso, prob_true_iso, iso_alpha, iso_beta),
                }

    cal_df = pd.DataFrame(calibration_records)
    cal_df.to_csv(out_dir / "calibration_comparison.csv", index=False)
    print(f"\nSaved calibration comparison table to {out_dir / 'calibration_comparison.csv'}")

    # Plot reliability diagrams (Seed 42)
    fig, axes = plt.subplots(1, len(models_info), figsize=(22, 4.5), sharey=True)
    for idx, (m_name, *_) in enumerate(models_info):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        c_uncal = curve_data[m_name]["uncal"]
        c_platt = curve_data[m_name]["platt"]
        c_iso = curve_data[m_name]["iso"]
        ax.plot(c_uncal[0], c_uncal[1], "s-", color="#E74C3C", label=f"Uncal (Slope={c_uncal[3]:.2f})")
        ax.plot(c_platt[0], c_platt[1], "o-", color="#2980B9", label=f"Platt (Slope={c_platt[3]:.2f})")
        ax.plot(c_iso[0], c_iso[1], "^-", color="#27AE60", label=f"Isotonic (Slope={c_iso[3]:.2f})")
        ax.set_title(m_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        if idx == 0:
            ax.set_ylabel("Fraction of Positives")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=7)

    plt.suptitle("Reliability Diagrams & Calibration Slopes (Seed 42, Untouched Test Set)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagrams.png", dpi=200)
    plt.close()
    print(f"Saved reliability diagrams to {out_dir / 'reliability_diagrams.png'}")

    # ================= STATISTICAL COMPARISONS =================
    print("\n=== PAIRED STATISTICAL COMPARISON ACROSS SEEDS (NADEAU-BENGIO CORRECTED) ===")
    piv = cal_df.pivot(index="seed", columns="model", values="pr_auc")

    # Paired differences
    diff_fggn_vs_uniform = piv["feature_group_gated"] - piv["uniform_group_net"]
    diff_fggn_vs_enscvdd = piv["feature_group_gated"] - piv["enscvdd_net"]
    diff_uniform_vs_enscvdd = piv["uniform_group_net"] - piv["enscvdd_net"]
    diff_fggn_vs_hgb = piv["feature_group_gated"] - piv["hist_gradient_boosting"]
    diff_uniform_vs_hgb = piv["uniform_group_net"] - piv["hist_gradient_boosting"]

    diff_df = pd.DataFrame({
        "seed": piv.index,
        "delta_fggn_minus_uniform": diff_fggn_vs_uniform,
        "delta_fggn_minus_enscvdd": diff_fggn_vs_enscvdd,
        "delta_uniform_minus_enscvdd": diff_uniform_vs_enscvdd,
        "delta_fggn_minus_hgb": diff_fggn_vs_hgb,
        "delta_uniform_minus_hgb": diff_uniform_vs_hgb,
    })
    diff_df.to_csv(out_dir / "paired_differences.csv", index=False)

    # Nadeau & Bengio (2003) parameters:
    # K = 5 splits, n_test = 38052, n_train = 177576 (70% train, 15% test)
    n_test = 38052
    n_train = 177576
    k_splits = len(diff_df)
    # Correction factor for repeated holdout splits sharing training data
    nb_factor = (1.0 / k_splits) + (n_test / n_train)

    def stats_summary(diff_series: pd.Series, name: str):
        exploratory = float(diff_series.loc[42])
        confirmatory = diff_series.drop(42)
        n = len(diff_series)
        mean = float(diff_series.mean())
        std = float(diff_series.std())
        se_standard = std / np.sqrt(n)

        # Standard uncorrected paired t-interval (df = n - 1 = 4)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower_std = mean - t_crit * se_standard
        ci_upper_std = mean + t_crit * se_standard
        p_val_standard = float(stats.ttest_1samp(diff_series, popmean=0.0).pvalue)

        # Nadeau & Bengio (2003) corrected resampled variance:
        # sigma^2_corr = (1/K + n_test/n_train) * S^2
        se_corrected = std * np.sqrt(nb_factor)
        ci_lower_corrected = mean - t_crit * se_corrected
        ci_upper_corrected = mean + t_crit * se_corrected
        t_stat_corrected = mean / se_corrected if se_corrected > 0 else 0.0
        p_val_corrected = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat_corrected), df=n - 1)))

        return {
            "comparison": name,
            "exploratory_seed_42_diff": exploratory,
            "confirmatory_mean_diff (seeds 101, 2024, 777, 9999)": float(confirmatory.mean()),
            "overall_mean_diff (5 seeds)": mean,
            "std_diff": std,
            "ci_95_lower_standard": ci_lower_std,
            "ci_95_upper_standard": ci_upper_std,
            "paired_t_pvalue_standard": p_val_standard,
            "se_nadeau_bengio": se_corrected,
            "ci_95_lower_nadeau_bengio": ci_lower_corrected,
            "ci_95_upper_nadeau_bengio": ci_upper_corrected,
            "pvalue_nadeau_bengio": p_val_corrected,
        }

    stat_rows = [
        stats_summary(diff_fggn_vs_uniform, "FGGN vs. UniformGroupNet (Learned Gate Value)"),
        stats_summary(diff_fggn_vs_enscvdd, "FGGN vs. EnsCVDD-Net (Grouped vs. 1D Sequence Hybrid)"),
        stats_summary(diff_uniform_vs_enscvdd, "UniformGroupNet vs. EnsCVDD-Net (Grouped vs. 1D Sequence)"),
        stats_summary(diff_fggn_vs_hgb, "FGGN vs. HistGradientBoosting (Neural vs. Tabular Tree)"),
        stats_summary(diff_uniform_vs_hgb, "UniformGroupNet vs. HistGradientBoosting"),
    ]

    stat_summary_df = pd.DataFrame(stat_rows)
    stat_summary_df.to_csv(out_dir / "statistical_paired_summary.csv", index=False)
    print("\nPaired PR-AUC Differences Table (with Nadeau-Bengio Correction):")
    print(stat_summary_df[["comparison", "overall_mean_diff (5 seeds)", "ci_95_lower_nadeau_bengio", "ci_95_upper_nadeau_bengio", "pvalue_nadeau_bengio"]].to_string(index=False))

    # Mean Calibration Summary
    print("\n=== MEAN CALIBRATION SUMMARY ACROSS 5 SEEDS ===")
    cal_mean = cal_df.groupby("model")[[
        "baseline_brier", "uncalibrated_brier", "platt_brier", "isotonic_brier",
        "uncalibrated_logloss", "platt_logloss", "isotonic_logloss",
        "uncalibrated_intercept", "platt_intercept", "isotonic_intercept",
        "uncalibrated_slope", "platt_slope", "isotonic_slope"
    ]].mean()
    cal_mean.to_csv(out_dir / "mean_calibration_summary.csv")
    print(cal_mean.to_string())


if __name__ == "__main__":
    main()

