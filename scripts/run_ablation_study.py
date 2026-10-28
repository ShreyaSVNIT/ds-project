#!/usr/bin/env python3
"""Ablation study script for Cardiovascular Disease Detection research.
Evaluates:
1. Imbalance strategy: none (cost-weighting) vs ADASYN vs random oversampling.
2. Feature selection: PBCC top-k (10, 15, 21).
3. Group Architecture: Learned softmax gating (FGGN) vs Uniform group pooling.
4. Thresholding: Fixed 0.50 vs Validation recall-constrained (>= 0.80).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.torch_models import FeatureGroupGatedNet, UniformGroupNet


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def oversample(X, y, method, seed):
    if method == "none":
        return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)
    from imblearn.over_sampling import ADASYN, RandomOverSampler
    sampler = ADASYN(random_state=seed, n_neighbors=5) if method == "adasyn" else RandomOverSampler(random_state=seed)
    Xr, yr = sampler.fit_resample(X, y)
    return np.asarray(Xr, dtype=np.float32), np.asarray(yr, dtype=np.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(yb)
        count += len(yb)
    return total_loss / max(count, 1)


def predict_prob(model, X, device, batch_size=4096):
    model.eval()
    probs = []
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            probs.append(torch.sigmoid(model(xb.to(device))).detach().cpu().numpy())
    return np.concatenate(probs)


def run_experiment(X, y, model_type, resampling, top_k, epochs, batch_size, lr, seed, device):
    from torch.utils.data import DataLoader, TensorDataset
    splits = make_splits(X, y, seed=seed)
    Xtr, Xv, Xte, features, _ = select_top_pbcc(splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k)

    scaler = StandardScaler().fit(Xtr)
    A_train = scaler.transform(Xtr).astype(np.float32)
    A_valid = scaler.transform(Xv).astype(np.float32)
    A_test = scaler.transform(Xte).astype(np.float32)

    A_fit, y_fit = oversample(pd.DataFrame(A_train, columns=features), splits.y_train, resampling, seed)
    y_valid = splits.y_valid.to_numpy(dtype=np.float32)
    y_test = splits.y_test.to_numpy(dtype=np.int32)

    torch.manual_seed(seed)
    if model_type == "fggn":
        model = FeatureGroupGatedNet(features).to(device)
    elif model_type == "uniform":
        model = UniformGroupNet(features).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    dataset = TensorDataset(torch.from_numpy(A_fit), torch.from_numpy(y_fit))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = max(float((y_fit == 0).sum()), 1.0) / max(float((y_fit == 1).sum()), 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    start = time.time()
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, criterion, device)
    duration = time.time() - start

    pv = predict_prob(model, A_valid, device)
    pt = predict_prob(model, A_test, device)

    threshold = choose_threshold_for_recall(splits.y_valid, pv, minimum_recall=0.80)
    records = []
    for t_name, t_val in [("fixed_0.50", 0.5), ("validation_recall_constrained", threshold)]:
        m = metric_dict(y_test, pt, threshold=t_val)
        records.append({
            "model_type": model_type,
            "resampling": resampling,
            "top_k": top_k,
            "seed": seed,
            "threshold_policy": t_name,
            "threshold": float(t_val),
            "fit_seconds": round(duration, 3),
            **m,
        })
    return records


def main():
    p = argparse.ArgumentParser(description="Run systematic ablation matrix.")
    p.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    p.add_argument("--out", default="artifacts/ablations")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")

    X, y = load_dataset(args.data)
    results = []

    # 1. Gated vs Uniform Ablation (with resampling none, top_k=21)
    print("\n=== Running Gated vs Uniform Ablation ===")
    for m_type in ["fggn", "uniform"]:
        print(f"  Running model_type={m_type}...")
        recs = run_experiment(X, y, model_type=m_type, resampling="none", top_k=21,
                              epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, seed=args.seed, device=device)
        results.extend(recs)

    # 2. Imbalance Handling Ablation (FGGN, top_k=21, resampling in [none, adasyn, random])
    print("\n=== Running Imbalance Strategy Ablation ===")
    for resamp in ["adasyn", "random"]:
        print(f"  Running resampling={resamp}...")
        recs = run_experiment(X, y, model_type="fggn", resampling=resamp, top_k=21,
                              epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, seed=args.seed, device=device)
        results.extend(recs)

    # 3. PBCC Feature Selection Ablation (FGGN, resampling none, top_k in [10, 15])
    print("\n=== Running PBCC Feature Selection Ablation ===")
    for k in [10, 15]:
        print(f"  Running top_k={k}...")
        recs = run_experiment(X, y, model_type="fggn", resampling="none", top_k=k,
                              epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, seed=args.seed, device=device)
        results.extend(recs)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "ablation_results.csv", index=False)
    print(f"\nAblations complete! Saved {len(results_df)} result rows to {out / 'ablation_results.csv'}")
    summary_cols = ["model_type", "resampling", "top_k", "threshold_policy", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc_average_precision"]
    print(results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
