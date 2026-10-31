#!/usr/bin/env python3
"""Multi-seed robustness study and per-example prediction persistence.
Evaluates:
- HistGradientBoosting (Tabular benchmark)
- LogisticRegression (Class-weighted linear baseline)
- FeatureGroupGatedNet (FGGN - Proposed domain-gated network)
- UniformGroupNet (Uniform group pooling ablation)
- EnsCVDDNet (Paper's LeNet1D + GRU parallel hybrid)

Runs across 5 fixed seeds: 42 (exploratory), 101, 2024, 777, 9999 (confirmatory).
Strictly fits scalers and feature selection on training data only.
Saves per-example test predictions to `artifacts/robustness/test_predictions.csv.gz`.
Saves run metadata, checkpoints, and summary tables with mean and standard deviation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import TARGET, load_dataset, make_group_splits, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.torch_models import EnsCVDDNet, FeatureGroupGatedNet, UniformGroupNet


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_torch_model(model, X_train, y_train, device, epochs=5, batch_size=1024, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = max(float((y_train == 0).sum()), 1.0) / max(float((y_train == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
    return model


def predict_torch_prob(model, X, device, batch_size=4096):
    model.eval()
    probs = []
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            probs.append(torch.sigmoid(model(xb.to(device))).detach().cpu().numpy())
    return np.concatenate(probs)


def run_seed_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    device: torch.device,
    epochs: int = 5,
    quick: bool = False,
    out_models_dir: Path | None = None,
    split_strategy: str = "row",
    target_recalls: tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.95),
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    split_fn = make_splits if split_strategy == "row" else make_group_splits
    splits = split_fn(X, y, seed=seed)
    Xtr, Xv, Xte, features, _ = select_top_pbcc(splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21)

    # Standardization strictly on training data
    scaler = StandardScaler().fit(Xtr)
    A_train = scaler.transform(Xtr).astype(np.float32)
    A_valid = scaler.transform(Xv).astype(np.float32)
    A_test = scaler.transform(Xte).astype(np.float32)

    y_train = splits.y_train.to_numpy(dtype=np.float32)
    y_valid = splits.y_valid.to_numpy(dtype=np.float32)
    y_test = splits.y_test.to_numpy(dtype=np.int32)
    test_indices = splits.test_indices

    model_metrics = []
    prediction_records = []
    validation_records = []
    threshold_records = []

    def persist_predictions(model_name, pv, pt, default_threshold):
        for idx, y_true, prob in zip(splits.valid_indices, y_valid, pv):
            validation_records.append({
                "seed": seed, "model": model_name, "split": "validation",
                "row_idx": int(idx), "y_true": int(y_true), "y_prob": float(prob),
            })
        for target_recall in target_recalls:
            threshold = choose_threshold_for_recall(
                y_valid, pv, minimum_recall=target_recall
            )
            metrics = metric_dict(y_test, pt, threshold)
            threshold_records.append({
                "seed": seed, "model": model_name,
                "target_recall": target_recall,
                "validation_threshold": threshold,
                "validation_recall": metric_dict(y_valid, pv, threshold)["recall_sensitivity"],
                **{f"test_{key}": value for key, value in metrics.items()},
            })

    # 1. Logistic Regression (balanced)
    print(f"  [Seed {seed}] Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed, solver="lbfgs")
    lr_model.fit(A_train, y_train)
    lr_dur = time.time() - t0
    pv_lr = lr_model.predict_proba(A_valid)[:, 1]
    pt_lr = lr_model.predict_proba(A_test)[:, 1]
    th_lr = choose_threshold_for_recall(y_valid, pv_lr, minimum_recall=0.80)
    persist_predictions("logistic_regression", pv_lr, pt_lr, th_lr)

    for p_name, p_thresh in [("fixed_0.50", 0.5), ("validation_recall_constrained", th_lr)]:
        m = metric_dict(y_test, pt_lr, p_thresh)
        model_metrics.append({
            "seed": seed, "model": "logistic_regression", "threshold_policy": p_name,
            "threshold": float(p_thresh), "fit_seconds": round(lr_dur, 3), **m,
        })
        if p_name == "validation_recall_constrained":
            for idx, y_true, prob in zip(test_indices, y_test, pt_lr):
                prediction_records.append({
                    "seed": seed, "model": "logistic_regression", "split": "test",
                    "row_idx": int(idx), "y_true": int(y_true), "y_prob": float(prob),
                    "threshold_policy": p_name, "threshold_value": float(p_thresh),
                    "y_pred": int(prob >= p_thresh),
                })
    if out_models_dir:
        joblib.dump(lr_model, out_models_dir / f"logistic_regression_seed_{seed}.joblib")

    # 2. HistGradientBoosting
    print(f"  [Seed {seed}] Training HistGradientBoosting...")
    t0 = time.time()
    # Cost weighting on positive class
    pos_ratio = max(float((y_train == 0).sum()), 1.0) / max(float((y_train == 1).sum()), 1.0)
    sw_train = np.where(y_train == 1, pos_ratio, 1.0)
    hgb_model = HistGradientBoostingClassifier(
        max_iter=80 if quick else 150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )
    hgb_model.fit(A_train, y_train, sample_weight=sw_train)
    hgb_dur = time.time() - t0
    pv_hgb = hgb_model.predict_proba(A_valid)[:, 1]
    pt_hgb = hgb_model.predict_proba(A_test)[:, 1]
    th_hgb = choose_threshold_for_recall(y_valid, pv_hgb, minimum_recall=0.80)
    persist_predictions("hist_gradient_boosting", pv_hgb, pt_hgb, th_hgb)

    for p_name, p_thresh in [("fixed_0.50", 0.5), ("validation_recall_constrained", th_hgb)]:
        m = metric_dict(y_test, pt_hgb, p_thresh)
        model_metrics.append({
            "seed": seed, "model": "hist_gradient_boosting", "threshold_policy": p_name,
            "threshold": float(p_thresh), "fit_seconds": round(hgb_dur, 3), **m,
        })
        if p_name == "validation_recall_constrained":
            for idx, y_true, prob in zip(test_indices, y_test, pt_hgb):
                prediction_records.append({
                    "seed": seed, "model": "hist_gradient_boosting", "split": "test",
                    "row_idx": int(idx), "y_true": int(y_true), "y_prob": float(prob),
                    "threshold_policy": p_name, "threshold_value": float(p_thresh),
                    "y_pred": int(prob >= p_thresh),
                })
    if out_models_dir:
        joblib.dump(hgb_model, out_models_dir / f"hist_gradient_boosting_seed_{seed}.joblib")

    # Neural Models factory
    neural_models = {
        "feature_group_gated": lambda: FeatureGroupGatedNet(features),
        "uniform_group_net": lambda: UniformGroupNet(features),
        "enscvdd_net": lambda: EnsCVDDNet(len(features)),
    }

    for name, factory in neural_models.items():
        print(f"  [Seed {seed}] Training {name}...")
        t0 = time.time()
        m_inst = factory()
        train_torch_model(m_inst, A_train, y_train, device, epochs=epochs, batch_size=1024, lr=1e-3, seed=seed)
        n_dur = time.time() - t0
        pv = predict_torch_prob(m_inst, A_valid, device)
        pt = predict_torch_prob(m_inst, A_test, device)
        th = choose_threshold_for_recall(y_valid, pv, minimum_recall=0.80)
        persist_predictions(name, pv, pt, th)

        for p_name, p_thresh in [("fixed_0.50", 0.5), ("validation_recall_constrained", th)]:
            m = metric_dict(y_test, pt, p_thresh)
            model_metrics.append({
                "seed": seed, "model": name, "threshold_policy": p_name,
                "threshold": float(p_thresh), "fit_seconds": round(n_dur, 3), **m,
            })
            if p_name == "validation_recall_constrained":
                for idx, y_true, prob in zip(test_indices, y_test, pt):
                    prediction_records.append({
                        "seed": seed, "model": name, "split": "test",
                        "row_idx": int(idx), "y_true": int(y_true), "y_prob": float(prob),
                        "threshold_policy": p_name, "threshold_value": float(p_thresh),
                        "y_pred": int(prob >= p_thresh),
                    })
        if out_models_dir:
            torch.save({"state_dict": m_inst.state_dict(), "features": features, "seed": seed},
                       out_models_dir / f"{name}_seed_{seed}.pt")

    return model_metrics, prediction_records, validation_records, threshold_records


def main():
    p = argparse.ArgumentParser(description="Multi-seed robustness evaluation across 5 seeds.")
    p.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    p.add_argument("--out", default="artifacts/robustness")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 777, 9999])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--split-strategy", choices=["row", "duplicate_group"], default="row")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    device = choose_device()
    print(f"Device: {device}")
    print(f"Evaluating across seeds: {args.seeds} (Seed 42 = exploratory; remaining = confirmatory)")

    X, y = load_dataset(args.data)

    all_metrics = []
    all_predictions = []
    all_validation_predictions = []
    all_threshold_sweeps = []

    for seed in args.seeds:
        print(f"\n=================== RUNNING SEED {seed} ===================")
        m_list, p_list, v_list, sweep_list = run_seed_evaluation(
            X, y, seed=seed, device=device, epochs=args.epochs, quick=args.quick,
            out_models_dir=models_dir, split_strategy=args.split_strategy
        )
        all_metrics.extend(m_list)
        all_predictions.extend(p_list)
        all_validation_predictions.extend(v_list)
        all_threshold_sweeps.extend(sweep_list)

    # Save all raw metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "all_seeds_results.csv", index=False)

    # Save compressed per-example test predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_path = out_dir / "test_predictions.csv.gz"
    pred_df.to_csv(pred_path, index=False, compression="gzip")
    print(f"\nSaved {len(pred_df):,} per-example test predictions to {pred_path}")
    pd.DataFrame(all_validation_predictions).to_csv(
        out_dir / "validation_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(all_threshold_sweeps).to_csv(
        out_dir / "threshold_sweep.csv", index=False
    )

    # Compute and format Mean ± Std summary
    constrained = metrics_df[metrics_df["threshold_policy"] == "validation_recall_constrained"].copy()
    grouped = constrained.groupby("model")

    summary_rows = []
    for model_name, grp in grouped:
        summary_rows.append({
            "model": model_name,
            "n_seeds": len(grp),
            "pr_auc_mean": grp["pr_auc_average_precision"].mean(),
            "pr_auc_std": grp["pr_auc_average_precision"].std(),
            "roc_auc_mean": grp["roc_auc"].mean(),
            "roc_auc_std": grp["roc_auc"].std(),
            "brier_score_mean": grp["brier_score"].mean(),
            "brier_score_std": grp["brier_score"].std(),
            "recall_mean": grp["recall_sensitivity"].mean(),
            "recall_std": grp["recall_sensitivity"].std(),
            "precision_mean": grp["precision"].mean(),
            "precision_std": grp["precision"].std(),
            "specificity_mean": grp["specificity"].mean(),
            "specificity_std": grp["specificity"].std(),
            "f1_mean": grp["f1"].mean(),
            "f1_std": grp["f1"].std(),
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="pr_auc_mean", ascending=False)
    summary_df.to_csv(out_dir / "robustness_summary.csv", index=False)

    # Save run metadata
    with (out_dir / "robustness_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seeds": args.seeds,
            "exploratory_seed": 42,
            "confirmatory_seeds": [s for s in args.seeds if s != 42],
            "epochs": args.epochs,
            "device": str(device),
            "target_recall": 0.80,
            "split_strategy": args.split_strategy,
            "n_total_rows": len(X),
            "methodology_note": "Repeated holdout splits share data overlap; variation reflects split sensitivity, not independent samples."
        }, f, indent=2)

    print("\n=== MULTI-SEED ROBUSTNESS SUMMARY (MEAN ± STD ACROSS 5 SEEDS) ===")
    display_cols = ["model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std", "recall_mean", "precision_mean", "specificity_mean", "brier_score_mean"]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
