#!/usr/bin/env python3
"""Run and evaluate InstanceGatedNet alongside FGGN and UniformGroupNet."""
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.torch_models import FeatureGroupGatedNet, InstanceGatedNet, UniformGroupNet

OUT = ROOT / "artifacts" / "instance_gating"


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(by)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, X, device):
    model.eval()
    bx = torch.from_numpy(X).to(device)
    logits = model(bx)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading data for InstanceGatedNet benchmark (Seed {args.seed})...")
    X, y = load_dataset(ROOT / "data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    splits = make_splits(X, y, seed=args.seed)
    Xtr, Xv, Xte, features, _ = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, top_k=21
    )

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xv_s = scaler.transform(Xv).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    ytr = splits.y_train.values.astype(np.float32)
    yv = splits.y_valid.values.astype(np.float32)
    yte = splits.y_test.values.astype(np.float32)

    # Class-weighted loss
    pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    dataset = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    models_to_test = {
        "UniformGroupNet": UniformGroupNet(features),
        "FGGN_GlobalGated": FeatureGroupGatedNet(features),
        "InstanceGatedNet": InstanceGatedNet(features),
    }

    results = []
    for name, model in models_to_test.items():
        print(f"--- Training {name} ---")
        model.to(device)
        torch.manual_seed(args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        start = time.time()
        for epoch in range(args.epochs):
            loss = train_epoch(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={loss:.4f}")
        fit_sec = time.time() - start

        v_prob = predict_proba(model, Xv_s, device)
        te_prob = predict_proba(model, Xte_s, device)

        thresh = choose_threshold_for_recall(splits.y_valid, v_prob, minimum_recall=0.80)
        metrics = metric_dict(splits.y_test, te_prob, thresh)
        metrics_05 = metric_dict(splits.y_test, te_prob, 0.50)

        results.append({
            "model": name,
            "threshold_policy": "validation_recall_constrained",
            "threshold": thresh,
            "fit_seconds": round(fit_sec, 2),
            **metrics,
        })
        results.append({
            "model": name,
            "threshold_policy": "fixed_0.50",
            "threshold": 0.50,
            "fit_seconds": round(fit_sec, 2),
            **metrics_05,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT / "instance_gating_results.csv", index=False)

    print("\n=== INSTANCE GATING BENCHMARK ===")
    summary_cols = ["model", "threshold_policy", "recall_sensitivity", "precision", "specificity", "pr_auc_average_precision", "roc_auc"]
    print(df[summary_cols].to_string(index=False))

    summary = {
        "models_evaluated": list(models_to_test.keys()),
        "finding": (
            "InstanceGatedNet computes per-patient routing weights conditioned on the joint feature embeddings. "
            "Evaluated at recall >= 80% on untouched held-out test data."
        )
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
