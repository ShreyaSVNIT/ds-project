#!/usr/bin/env python3
"""Rigorous hyperparameter optimization for FGGN++ and HistGradientBoosting using Optuna.

Optimization criterion: Validation PR-AUC on the held-out validation set (seed 42).
The test set is STRICTLY UNTOUCHED during hyperparameter search.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.fggn_variants import FGGNPlusPlus, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Tune FGGN++ and HistGBT using Optuna.")
    parser.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    parser.add_argument("--out", default="artifacts/tuning")
    parser.add_argument("--neural-trials", type=int, default=15)
    parser.add_argument("--hgb-trials", type=int, default=25)
    parser.add_argument("--neural-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_eval_fggn(
    model: nn.Module,
    train_loader: DataLoader,
    X_val_tensor: torch.Tensor,
    y_val: np.ndarray,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> tuple[float, float]:
    """Train for fixed epochs, return final validation PR-AUC and ROC-AUC."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    return val_pr_auc, val_roc_auc


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print("======================================================================")
    print("  PHASE 3: RIGOROUS HYPERPARAMETER OPTIMIZATION (VALIDATION ONLY)    ")
    print("======================================================================")
    print(f"Loading data from {args.data}...")
    X, y = load_dataset(args.data)
    splits = make_splits(X, y, seed=args.seed)

    Xtr, Xv, Xte, features, _ = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21
    )

    scaler = StandardScaler().fit(Xtr)
    A_train = scaler.transform(Xtr).astype(np.float32)
    A_valid = scaler.transform(Xv).astype(np.float32)
    # NOTE: A_test is strictly ignored during optimization!

    y_train = splits.y_train.to_numpy(dtype=np.float32)
    y_valid = splits.y_valid.to_numpy(dtype=np.float32)

    pos_ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    print(f"Train size: {len(y_train):,} | Valid size: {len(y_valid):,} | Imbalance pos_ratio: {pos_ratio:.2f}")

    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    X_val_t = torch.from_numpy(A_valid)

    # ------------------------------------------------------------------
    # 1. Tune FGGN++
    # ------------------------------------------------------------------
    print(f"\n--- Starting Optuna Study for FGGN++ ({args.neural_trials} trials, {args.neural_epochs} epochs each) ---")
    neural_trial_records = []

    def neural_objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
        embed_dim = trial.suggest_categorical("embed_dim", [8, 16])
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048])
        init_temp = trial.suggest_float("initial_temperature", 0.5, 2.0)
        loss_type = trial.suggest_categorical("loss_type", ["weighted_bce", "focal"])

        if loss_type == "focal":
            gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
            alpha = trial.suggest_float("focal_alpha", 0.25, 0.75)
            criterion = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_ratio)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_ratio]))

        torch.manual_seed(args.seed + trial.number)
        model = FGGNPlusPlus(
            features,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=2,
            initial_temperature=init_temp,
            dropout=dropout,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

        dataset = TensorDataset(torch.from_numpy(A_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        t0 = time.time()
        val_pr, val_roc = train_eval_fggn(
            model, loader, X_val_t, y_valid, criterion, optimizer, args.neural_epochs, device
        )
        elapsed = time.time() - t0

        trial_rec = {
            "trial": trial.number,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "lr": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "initial_temperature": init_temp,
            "loss_type": loss_type,
            "val_pr_auc": val_pr,
            "val_roc_auc": val_roc,
            "elapsed_seconds": round(elapsed, 2),
        }
        neural_trial_records.append(trial_rec)
        print(f"  [Neural Trial {trial.number:02d}] PR-AUC: {val_pr:.4f} | ROC-AUC: {val_roc:.4f} | time: {elapsed:.1f}s")
        return val_pr

    neural_sampler = TPESampler(seed=args.seed)
    neural_study = optuna.create_study(direction="maximize", sampler=neural_sampler)
    neural_study.optimize(neural_objective, n_trials=args.neural_trials)

    print(f"\nBest FGGN++ Validation PR-AUC: {neural_study.best_value:.4f}")
    print(f"Best FGGN++ Parameters: {neural_study.best_params}")

    pd.DataFrame(neural_trial_records).to_csv(out_dir / "trials_neural.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Tune HistGradientBoosting under identical validation protocol
    # ------------------------------------------------------------------
    print(f"\n--- Starting Optuna Study for HistGradientBoosting ({args.hgb_trials} trials) ---")
    hgb_trial_records = []

    def hgb_objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.03, 0.15)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 60)
        l2_reg = trial.suggest_float("l2_regularization", 0.01, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 80, 200)
        use_sample_weight = trial.suggest_categorical("use_sample_weight", [True, False])

        sw = np.where(y_train == 1, pos_ratio, 1.0) if use_sample_weight else None

        hgb = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            max_iter=max_iter,
            random_state=args.seed + trial.number,
        )

        t0 = time.time()
        hgb.fit(A_train, y_train, sample_weight=sw)
        elapsed = time.time() - t0

        val_probs = hgb.predict_proba(A_valid)[:, 1]
        val_pr = float(average_precision_score(y_valid, val_probs))
        val_roc = float(roc_auc_score(y_valid, val_probs))

        rec = {
            "trial": trial.number,
            "learning_rate": lr,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_reg,
            "max_iter": max_iter,
            "use_sample_weight": use_sample_weight,
            "val_pr_auc": val_pr,
            "val_roc_auc": val_roc,
            "elapsed_seconds": round(elapsed, 2),
        }
        hgb_trial_records.append(rec)
        print(f"  [HGB Trial {trial.number:02d}] PR-AUC: {val_pr:.4f} | ROC-AUC: {val_roc:.4f} | time: {elapsed:.1f}s")
        return val_pr

    hgb_sampler = TPESampler(seed=args.seed)
    hgb_study = optuna.create_study(direction="maximize", sampler=hgb_sampler)
    hgb_study.optimize(hgb_objective, n_trials=args.hgb_trials)

    print(f"\nBest HistGradientBoosting Validation PR-AUC: {hgb_study.best_value:.4f}")
    print(f"Best HistGradientBoosting Parameters: {hgb_study.best_params}")

    pd.DataFrame(hgb_trial_records).to_csv(out_dir / "trials_hgb.csv", index=False)

    best_payload = {
        "validation_protocol": "Stratified 15% validation split (seed 42); zero test set leakage.",
        "fggn_plus_plus": {
            "best_validation_pr_auc": float(neural_study.best_value),
            "best_params": neural_study.best_params,
        },
        "hist_gradient_boosting": {
            "best_validation_pr_auc": float(hgb_study.best_value),
            "best_params": hgb_study.best_params,
        },
    }
    with (out_dir / "best_hparams.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print(f"\nOptimization results saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
