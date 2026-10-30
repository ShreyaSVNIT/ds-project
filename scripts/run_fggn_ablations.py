#!/usr/bin/env python3
"""Comprehensive Controlled Ablation Study & Multi-Seed Benchmark.

Compares across 5 fixed random seeds (42, 101, 2024, 777, 9999):
1. EnsCVDD-Net (Paper 1D CNN + GRU baseline)
2. FeatureGroupGatedNet (Existing FGGN baseline)
3. UniformGroupNet (Equal group pooling ablation)
4. Residual FGGN (Ablation 1: ResBlocks + LayerNorm + GELU)
5. Feature-wise Attention FGGN (Ablation 2: Intra-Group Attention)
6. Cross-Group Interaction FGGN (Ablation 3: Second-Order Pairwise Interactions)
7. Hierarchical Gating FGGN (Ablation 4: Feature Gate + Calibrated Group Gate)
8. Categorical Embedding FGGN (Ablation 5: Entity Embeddings for Ordinals)
9. FGGN++ (Flagship unified architecture with Focal Loss)
10. HistGradientBoosting (Baseline setting)
11. HistGradientBoosting (Optuna-tuned setting)

Guarantees:
- Naturally-imbalanced held-out test partitions strictly preserved (N=38,052 per seed).
- Zero test data leakage: scalers, feature ranking, threshold selection strictly on train/val.
- Computes PR-AUC, ROC-AUC, Brier score, ECE, Recall, Precision, Specificity, F1, Balanced Accuracy.
- Records parameter count, training time, and inference throughput (samples/sec).
- Saves full summary tables, paired statistical significance tests (t-test, Wilcoxon).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.torch_models import EnsCVDDNet, FeatureGroupGatedNet, UniformGroupNet
from cardio.fggn_variants import (
    CategoricalEmbeddingFGGN,
    CrossGroupInteractionFGGN,
    FeatureAttentionFGGN,
    FGGNPlusPlus,
    FocalLoss,
    HierarchicalGatingFGGN,
    ResidualFGGN,
)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (prob >= bin_limits[i]) & (prob < bin_limits[i + 1])
        if i == n_bins - 1:
            mask = mask | (prob == bin_limits[i + 1])
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(prob[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def count_parameters(model: Any) -> int:
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


def train_neural_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_fn: Optional[nn.Module] = None,
    pos_weight: float = 9.61,
) -> float:
    """Train PyTorch model for fixed epochs, return training duration in seconds."""
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_fn is None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = loss_fn.to(device)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
    return time.time() - t0


@torch.no_grad()
def predict_neural_prob(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096
) -> Tuple[np.ndarray, float]:
    """Inference over array X; returns predicted probabilities and inference duration."""
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_list = []
    t0 = time.time()
    for (bx,) in loader:
        logits = model(bx.to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
    dur = time.time() - t0
    return np.concatenate(probs_list), dur


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Controlled Ablation Benchmark.")
    parser.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--out", default="artifacts/ablations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 777, 9999])
    parser.add_argument("--neural-epochs", type=int, default=4)
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print("======================================================================")
    print("  PHASE 4 & 5: CONTROLLED ABLATIONS & MULTI-SEED RIGOROUS BENCHMARK   ")
    print("======================================================================")
    print(f"Dataset: {args.data}")
    print(f"Random Seeds: {args.seeds}")
    print(f"Epochs per neural model: {args.neural_epochs}")
    print(f"Output Directory: {out_dir.resolve()}\n")

    X, y = load_dataset(args.data)

    models_to_evaluate = [
        "enscvdd_net",
        "fggn_baseline",
        "uniform_group_net",
        "residual_fggn",
        "feature_attention_fggn",
        "cross_group_interaction_fggn",
        "hierarchical_gating_fggn",
        "categorical_embedding_fggn",
        "fggn_plus_plus",
        "hist_gradient_boosting",
        "hist_gradient_boosting_tuned",
    ]

    all_results: List[Dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n==================================================================")
        print(f"  EXECUTING SEED {seed} (Stratified 70/15/15 Split)")
        print(f"==================================================================")
        splits = make_splits(X, y, seed=seed)
        Xtr, Xv, Xte, features, _ = select_top_pbcc(
            splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21
        )

        scaler = StandardScaler().fit(Xtr)
        A_train = scaler.transform(Xtr).astype(np.float32)
        A_valid = scaler.transform(Xv).astype(np.float32)
        A_test = scaler.transform(Xte).astype(np.float32)

        y_train = splits.y_train.to_numpy(dtype=np.float32)
        y_valid = splits.y_valid.to_numpy(dtype=np.float32)
        y_test = splits.y_test.to_numpy(dtype=np.int32)

        pos_ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()

        for model_name in models_to_evaluate:
            print(f"  [Seed {seed}] Training & Evaluating: {model_name}...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            if model_name == "hist_gradient_boosting":
                # HistGradientBoosting baseline (from literature & original experiments)
                hgb = HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    max_iter=150,
                    random_state=seed,
                )
                sw = np.where(y_train == 1, pos_ratio, 1.0)
                t0 = time.time()
                hgb.fit(A_train, y_train, sample_weight=sw)
                fit_sec = time.time() - t0

                t0_inf = time.time()
                v_prob = hgb.predict_proba(A_valid)[:, 1]
                t_prob = hgb.predict_proba(A_test)[:, 1]
                inf_sec = time.time() - t0_inf
                param_count = 0
                joblib.dump(hgb, out_dir / "models" / f"hist_gradient_boosting_seed_{seed}.joblib")

            elif model_name == "hist_gradient_boosting_tuned":
                # Optuna-tuned HistGradientBoosting
                hgb = HistGradientBoostingClassifier(
                    learning_rate=0.134,
                    max_leaf_nodes=44,
                    min_samples_leaf=46,
                    l2_regularization=0.0115,
                    max_iter=197,
                    random_state=seed,
                )
                sw = np.where(y_train == 1, pos_ratio, 1.0)
                t0 = time.time()
                hgb.fit(A_train, y_train, sample_weight=sw)
                fit_sec = time.time() - t0

                t0_inf = time.time()
                v_prob = hgb.predict_proba(A_valid)[:, 1]
                t_prob = hgb.predict_proba(A_test)[:, 1]
                inf_sec = time.time() - t0_inf
                param_count = 0
                joblib.dump(hgb, out_dir / "models" / f"hist_gradient_boosting_tuned_seed_{seed}.joblib")

            else:
                # PyTorch Neural Models
                loss_fn = None
                lr = 1e-3
                weight_decay = 1e-5
                batch_size = 1024

                if model_name == "enscvdd_net":
                    net = EnsCVDDNet(len(features))
                elif model_name == "fggn_baseline":
                    net = FeatureGroupGatedNet(features, hidden_dim=16)
                elif model_name == "uniform_group_net":
                    net = UniformGroupNet(features, hidden_dim=16)
                elif model_name == "residual_fggn":
                    net = ResidualFGGN(features, hidden_dim=32, dropout=0.1)
                elif model_name == "feature_attention_fggn":
                    net = FeatureAttentionFGGN(features, embed_dim=16, hidden_dim=32, dropout=0.1)
                elif model_name == "cross_group_interaction_fggn":
                    net = CrossGroupInteractionFGGN(features, hidden_dim=32, dropout=0.1)
                elif model_name == "hierarchical_gating_fggn":
                    net = HierarchicalGatingFGGN(features, hidden_dim=32, initial_temperature=1.0, dropout=0.1)
                elif model_name == "categorical_embedding_fggn":
                    net = CategoricalEmbeddingFGGN(
                        features, embed_dim=16, hidden_dim=32, dropout=0.1,
                        scaler_mean=scaler_mean, scaler_scale=scaler_scale
                    )
                elif model_name == "fggn_plus_plus":
                    net = FGGNPlusPlus(
                        features,
                        embed_dim=8,
                        hidden_dim=32,
                        num_heads=2,
                        initial_temperature=1.0,
                        dropout=0.10,
                        scaler_mean=scaler_mean,
                        scaler_scale=scaler_scale,
                    )
                    lr = 0.001
                    weight_decay = 1e-5
                    loss_fn = FocalLoss(alpha=0.50, gamma=1.63, pos_weight=pos_ratio)
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                param_count = count_parameters(net)
                fit_sec = train_neural_model(
                    net,
                    A_train,
                    y_train,
                    device=device,
                    epochs=args.neural_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    loss_fn=loss_fn,
                    pos_weight=pos_ratio,
                )
                v_prob, _ = predict_neural_prob(net, A_valid, device=device)
                t_prob, inf_sec = predict_neural_prob(net, A_test, device=device)
                torch.save(
                    {"state_dict": net.state_dict(), "features": features, "seed": seed},
                    out_dir / "models" / f"{model_name}_seed_{seed}.pt",
                )

            # Operational threshold: chosen strictly on validation for recall >= 0.80
            thresh_recall = choose_threshold_for_recall(y_valid, v_prob, minimum_recall=0.80)
            val_recall_achieved = float(np.mean(v_prob[y_valid == 1] >= thresh_recall))
            val_prec_achieved = float(
                np.sum((v_prob >= thresh_recall) & (y_valid == 1)) / max(np.sum(v_prob >= thresh_recall), 1)
            )

            # Compute calibration ECE
            ece_val = expected_calibration_error(y_test, t_prob, n_bins=10)
            throughput = len(y_test) / max(inf_sec, 1e-4)

            # Metrics at validation recall >= 0.80 threshold
            m_recall = metric_dict(y_test, t_prob, thresh_recall)
            # Metrics at default 0.50 threshold
            m_05 = metric_dict(y_test, t_prob, 0.50)

            all_results.append({
                "seed": seed,
                "model": model_name,
                "threshold_policy": "validation_recall_constrained",
                "threshold": float(thresh_recall),
                "val_recall_achieved": round(val_recall_achieved, 4),
                "val_prec_achieved": round(val_prec_achieved, 4),
                "fit_seconds": round(fit_sec, 3),
                "inference_seconds": round(inf_sec, 4),
                "throughput_samples_per_sec": round(throughput, 1),
                "param_count": param_count,
                "ece": round(ece_val, 4),
                **m_recall,
            })
            all_results.append({
                "seed": seed,
                "model": model_name,
                "threshold_policy": "fixed_0.50",
                "threshold": 0.50,
                "val_recall_achieved": None,
                "val_prec_achieved": None,
                "fit_seconds": round(fit_sec, 3),
                "inference_seconds": round(inf_sec, 4),
                "throughput_samples_per_sec": round(throughput, 1),
                "param_count": param_count,
                "ece": round(ece_val, 4),
                **m_05,
            })

            print(
                f"    -> Test PR-AUC: {m_recall['pr_auc_average_precision']:.4f} | "
                f"ROC-AUC: {m_recall['roc_auc']:.4f} | "
                f"Precision@Recall>=80: {m_recall['precision']*100:.2f}% | "
                f"Fit Time: {fit_sec:.1f}s | Params: {param_count:,}"
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "all_seed_results.csv", index=False)

    # ------------------------------------------------------------------
    # Multi-Seed Aggregation & Summary Table
    # ------------------------------------------------------------------
    recall_df = results_df[results_df["threshold_policy"] == "validation_recall_constrained"]
    
    summary_rows = []
    for model_name in models_to_evaluate:
        sub = recall_df[recall_df["model"] == model_name]
        pr_vals = sub["pr_auc_average_precision"].values
        roc_vals = sub["roc_auc"].values
        prec_vals = sub["precision"].values
        rec_vals = sub["recall_sensitivity"].values
        spec_vals = sub["specificity"].values
        f1_vals = sub["f1"].values
        brier_vals = sub["brier_score"].values
        ece_vals = sub["ece"].values
        time_vals = sub["fit_seconds"].values
        params = int(sub["param_count"].iloc[0])

        summary_rows.append({
            "model": model_name,
            "pr_auc_mean": float(np.mean(pr_vals)),
            "pr_auc_std": float(np.std(pr_vals, ddof=1)),
            "roc_auc_mean": float(np.mean(roc_vals)),
            "roc_auc_std": float(np.std(roc_vals, ddof=1)),
            "precision_mean": float(np.mean(prec_vals)),
            "precision_std": float(np.std(prec_vals, ddof=1)),
            "recall_mean": float(np.mean(rec_vals)),
            "recall_std": float(np.std(rec_vals, ddof=1)),
            "specificity_mean": float(np.mean(spec_vals)),
            "specificity_std": float(np.std(spec_vals, ddof=1)),
            "f1_mean": float(np.mean(f1_vals)),
            "f1_std": float(np.std(f1_vals, ddof=1)),
            "brier_mean": float(np.mean(brier_vals)),
            "brier_std": float(np.std(brier_vals, ddof=1)),
            "ece_mean": float(np.mean(ece_vals)),
            "ece_std": float(np.std(ece_vals, ddof=1)),
            "fit_sec_mean": float(np.mean(time_vals)),
            "param_count": params,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="pr_auc_mean", ascending=False)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Statistical Significance Testing (Paired t-test & Wilcoxon)
    # ------------------------------------------------------------------
    hgb_base_prs = recall_df[recall_df["model"] == "hist_gradient_boosting"].sort_values("seed")["pr_auc_average_precision"].values
    hgb_tune_prs = recall_df[recall_df["model"] == "hist_gradient_boosting_tuned"].sort_values("seed")["pr_auc_average_precision"].values
    fggn_base_prs = recall_df[recall_df["model"] == "fggn_baseline"].sort_values("seed")["pr_auc_average_precision"].values
    fggn_plus_prs = recall_df[recall_df["model"] == "fggn_plus_plus"].sort_values("seed")["pr_auc_average_precision"].values

    # FGGN++ vs FGGN Baseline
    t_fggn_base, p_fggn_base = stats.ttest_rel(fggn_plus_prs, fggn_base_prs)
    w_fggn_base, p_w_fggn_base = stats.wilcoxon(fggn_plus_prs, fggn_base_prs)

    # FGGN++ vs HistGradientBoosting Baseline
    t_hgb_base, p_hgb_base = stats.ttest_rel(fggn_plus_prs, hgb_base_prs)
    w_hgb_base, p_w_hgb_base = stats.wilcoxon(fggn_plus_prs, hgb_base_prs)

    # FGGN++ vs HistGradientBoosting Tuned
    t_hgb_tune, p_hgb_tune = stats.ttest_rel(fggn_plus_prs, hgb_tune_prs)
    w_hgb_tune, p_w_hgb_tune = stats.wilcoxon(fggn_plus_prs, hgb_tune_prs)

    stat_summary = {
        "fggn_plus_plus_vs_fggn_baseline": {
            "fggn_plus_plus_pr_auc": f"{np.mean(fggn_plus_prs):.4f} +/- {np.std(fggn_plus_prs, ddof=1):.4f}",
            "fggn_baseline_pr_auc": f"{np.mean(fggn_base_prs):.4f} +/- {np.std(fggn_base_prs, ddof=1):.4f}",
            "delta_pr_auc": float(np.mean(fggn_plus_prs) - np.mean(fggn_base_prs)),
            "paired_t_stat": float(t_fggn_base),
            "paired_t_pvalue": float(p_fggn_base),
            "wilcoxon_stat": float(w_fggn_base),
            "wilcoxon_pvalue": float(p_w_fggn_base),
        },
        "fggn_plus_plus_vs_hist_gradient_boosting_baseline": {
            "fggn_plus_plus_pr_auc": f"{np.mean(fggn_plus_prs):.4f} +/- {np.std(fggn_plus_prs, ddof=1):.4f}",
            "hist_gradient_boosting_pr_auc": f"{np.mean(hgb_base_prs):.4f} +/- {np.std(hgb_base_prs, ddof=1):.4f}",
            "delta_pr_auc": float(np.mean(fggn_plus_prs) - np.mean(hgb_base_prs)),
            "paired_t_stat": float(t_hgb_base),
            "paired_t_pvalue": float(p_hgb_base),
            "wilcoxon_stat": float(w_hgb_base),
            "wilcoxon_pvalue": float(p_w_hgb_base),
        },
        "fggn_plus_plus_vs_hist_gradient_boosting_tuned": {
            "fggn_plus_plus_pr_auc": f"{np.mean(fggn_plus_prs):.4f} +/- {np.std(fggn_plus_prs, ddof=1):.4f}",
            "hist_gradient_boosting_tuned_pr_auc": f"{np.mean(hgb_tune_prs):.4f} +/- {np.std(hgb_tune_prs, ddof=1):.4f}",
            "delta_pr_auc": float(np.mean(fggn_plus_prs) - np.mean(hgb_tune_prs)),
            "paired_t_stat": float(t_hgb_tune),
            "paired_t_pvalue": float(p_hgb_tune),
            "wilcoxon_stat": float(w_hgb_tune),
            "wilcoxon_pvalue": float(p_w_hgb_tune),
        },
    }
    with (out_dir / "statistical_tests.json").open("w", encoding="utf-8") as f:
        json.dump(stat_summary, f, indent=2)

    print("\n======================================================================")
    print("  DEFINITIVE MULTI-SEED BENCHMARK SUMMARY (N=38,052 per seed)         ")
    print("======================================================================")
    formatted_summary = []
    for _, r in summary_df.iterrows():
        formatted_summary.append({
            "Model Architecture": r["model"],
            "PR-AUC (Mean ± Std)": f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
            "ROC-AUC (Mean ± Std)": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
            "Precision": f"{r['precision_mean']*100:.2f}%",
            "Recall": f"{r['recall_mean']*100:.2f}%",
            "Specificity": f"{r['specificity_mean']*100:.2f}%",
            "Brier": f"{r['brier_mean']:.4f}",
            "ECE": f"{r['ece_mean']:.4f}",
            "Params": f"{r['param_count']:,}",
            "Train Sec": f"{r['fit_sec_mean']:.1f}s",
        })
    print(pd.DataFrame(formatted_summary).to_string(index=False))

    print("\n======================================================================")
    print("  STATISTICAL SIGNIFICANCE COMPARISONS                                ")
    print("======================================================================")
    print(f"FGGN++ vs FGGN Baseline: Delta = {stat_summary['fggn_plus_plus_vs_fggn_baseline']['delta_pr_auc']:+.4f} "
          f"(t={t_fggn_base:.3f}, p={p_fggn_base:.4f})")
    print(f"FGGN++ vs HistGBT Baseline: Delta = {stat_summary['fggn_plus_plus_vs_hist_gradient_boosting_baseline']['delta_pr_auc']:+.4f} "
          f"(t={t_hgb_base:.3f}, p={p_hgb_base:.4f})")
    print(f"FGGN++ vs HistGBT Tuned: Delta = {stat_summary['fggn_plus_plus_vs_hist_gradient_boosting_tuned']['delta_pr_auc']:+.4f} "
          f"(t={t_hgb_tune:.3f}, p={p_hgb_tune:.4f})")
    print(f"\nAll artifacts successfully saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
