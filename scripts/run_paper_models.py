#!/usr/bin/env python3
"""Structural reimplementation of the paper's LeNet/GRU/Ensemble/Blending family."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import load_dataset, make_splits, select_top_pbcc
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.torch_models import EnsCVDDNet, FeatureGroupGatedNet, GRUClassifier, LeNet1D, MetaMLP, UniformGroupNet


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def oversample(X, y, method, seed):
    if method == "none":
        return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)
    sampler = ADASYN(random_state=seed, n_neighbors=5) if method == "adasyn" else RandomOverSampler(random_state=seed)
    Xr, yr = sampler.fit_resample(X, y)
    return np.asarray(Xr, dtype=np.float32), np.asarray(yr, dtype=np.float32)


def train_model(model, X, y, device, epochs, batch_size, lr, seed, positive_weight=None):
    """Train for a fixed epoch budget; external validation is kept for threshold/meta learning."""
    torch.manual_seed(seed)
    model.to(device)
    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if positive_weight is None:
        positive_weight = 1.0
    weight = torch.tensor(float(positive_weight), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += float(loss.detach()) * len(yb)
            count += len(yb)
        print(f"  epoch {epoch + 1:02d}/{epochs} train_loss={running / max(count,1):.4f}")
    return model


def predict_prob(model, X, device, batch_size=8192):
    model.eval()
    probs = []
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            probs.append(torch.sigmoid(model(xb.to(device))).detach().cpu().numpy())
    return np.concatenate(probs)


def predict_group_gates(model, X, device, batch_size=8192):
    model.eval()
    outputs = []
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            _, weights = model.forward_with_gates(xb.to(device))
            outputs.append(weights.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_meta(X_meta, y_meta, device, epochs, batch_size, seed):
    model = MetaMLP(n_inputs=X_meta.shape[1]).to(device)
    X = torch.from_numpy(np.asarray(X_meta, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_meta, dtype=np.float32))
    loader = DataLoader(TensorDataset(X, y), batch_size=min(batch_size, 2048), shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(seed)
    for _ in range(max(epochs, 5)):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/heart_disease_health_indicators_BRFSS2015.csv")
    p.add_argument("--out", default="artifacts/neural")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resampling", choices=["none", "adasyn", "random"], default="adasyn")
    p.add_argument("--top-k", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    X, y = load_dataset(args.data)
    splits = make_splits(X, y, seed=args.seed)
    Xtr, Xv, Xte, features, scores = select_top_pbcc(
        splits.X_train, splits.X_valid, splits.X_test, splits.y_train, args.top_k
    )
    # Standardization is fitted on training data only. It keeps the scale stable before ADASYN.
    scaler = StandardScaler().fit(Xtr)
    A_train = scaler.transform(Xtr).astype(np.float32)
    A_valid = scaler.transform(Xv).astype(np.float32)
    A_test = scaler.transform(Xte).astype(np.float32)
    A_fit, y_fit = oversample(pd.DataFrame(A_train, columns=features), splits.y_train, args.resampling, args.seed)
    y_valid = splits.y_valid.to_numpy(dtype=np.float32)
    y_test = splits.y_test.to_numpy(dtype=np.int32)

    model_specs = {
        "lenet1d": lambda: LeNet1D(A_train.shape[1]),
        "gru": lambda: GRUClassifier(),
        "enscvdd_style": lambda: EnsCVDDNet(A_train.shape[1]),
        "feature_group_gated": lambda: FeatureGroupGatedNet(features),
        "uniform_group_ablation": lambda: UniformGroupNet(features),
    }
    metrics = []
    probabilities = {}
    group_gate_summary = None
    for name, factory in model_specs.items():
        print(f"\nTraining {name}...")
        model = factory()
        start = time.time()
        pos = max(float((y_fit == 0).sum()), 1.0) / max(float((y_fit == 1).sum()), 1.0)
        # On a resampled training set this is about 1; otherwise it cost-weights the positive class.
        model = train_model(model, A_fit, y_fit, device, args.epochs, args.batch_size,
                            args.lr, args.seed, positive_weight=pos)
        duration = time.time() - start
        pv = predict_prob(model, A_valid, device)
        pt = predict_prob(model, A_test, device)
        probabilities[name] = {"valid": pv, "test": pt}
        if name == "feature_group_gated":
            gate_weights = predict_group_gates(model, A_test, device)
            group_gate_summary = {group: float(gate_weights[:, i].mean())
                                  for i, group in enumerate(model.group_names)}
        threshold = choose_threshold_for_recall(splits.y_valid, pv, minimum_recall=0.80)
        val_recall_at_thresh = float(np.mean(pv[y_valid == 1] >= threshold))
        val_prec_at_thresh = float(np.sum((pv >= threshold) & (y_valid == 1)) / max(np.sum(pv >= threshold), 1))
        for threshold_name, operating_threshold in [("fixed_0.50", 0.5), ("validation_recall_constrained", threshold)]:
            metrics.append({
                "model": name, "threshold_policy": threshold_name,
                "threshold": float(operating_threshold),
                "val_recall_achieved": round(val_recall_at_thresh, 4) if threshold_name != "fixed_0.50" else None,
                "val_prec_achieved": round(val_prec_at_thresh, 4) if threshold_name != "fixed_0.50" else None,
                "fit_seconds": round(duration, 3),
                "resampling": args.resampling,
                **metric_dict(y_test, pt, operating_threshold)
            })
        save_payload = {"state_dict": model.state_dict(), "model_name": name, "n_features": A_train.shape[1],
                        "features": features, "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist()}
        if name == "feature_group_gated":
            save_payload["group_names"] = list(model.group_names)
            save_payload["group_indices"] = [list(idx) for idx in model.group_indices]
        torch.save(save_payload, out / "models" / f"{name}.pt")

    # Blending: base models' validation predictions train the MLP meta-learner.
    print("\nTraining BlCVDD-style MLP on one half of validation predictions...")
    from sklearn.model_selection import train_test_split
    valid_idx = np.arange(len(y_valid))
    meta_idx, threshold_idx = train_test_split(valid_idx, test_size=0.5, random_state=args.seed, stratify=y_valid)
    meta_valid_features = np.column_stack([probabilities["lenet1d"]["valid"][meta_idx],
                                           probabilities["gru"]["valid"][meta_idx]]).astype(np.float32)
    threshold_valid_features = np.column_stack([probabilities["lenet1d"]["valid"][threshold_idx],
                                                probabilities["gru"]["valid"][threshold_idx]]).astype(np.float32)
    meta_start = time.time()
    meta = fit_meta(meta_valid_features, y_valid[meta_idx], device, args.epochs, args.batch_size, args.seed)
    meta_duration = time.time() - meta_start
    meta_threshold_prob = predict_prob(meta, threshold_valid_features, device)
    meta_test = np.column_stack([probabilities["lenet1d"]["test"], probabilities["gru"]["test"]]).astype(np.float32)
    meta_prob = predict_prob(meta, meta_test, device)
    meta_threshold = choose_threshold_for_recall(y_valid[threshold_idx], meta_threshold_prob, minimum_recall=0.80)
    meta_val_recall = float(np.mean(meta_threshold_prob[y_valid[threshold_idx] == 1] >= meta_threshold))
    meta_val_prec = float(np.sum((meta_threshold_prob >= meta_threshold) & (y_valid[threshold_idx] == 1)) / max(np.sum(meta_threshold_prob >= meta_threshold), 1))
    for threshold_name, operating_threshold in [("fixed_0.50", 0.5), ("validation_recall_constrained", meta_threshold)]:
        metrics.append({
            "model": "blcvdd_style_validation_blend",
            "threshold_policy": threshold_name,
            "threshold": float(operating_threshold),
            "val_recall_achieved": round(meta_val_recall, 4) if threshold_name != "fixed_0.50" else None,
            "val_prec_achieved": round(meta_val_prec, 4) if threshold_name != "fixed_0.50" else None,
            "fit_seconds": round(meta_duration, 3),
            "resampling": args.resampling,
            **metric_dict(y_test, meta_prob, operating_threshold)
        })
    torch.save({"state_dict": meta.state_dict(), "features": features, "meta_indices": meta_idx.tolist(),
                "threshold_indices": threshold_idx.tolist()}, out / "models" / "blcvdd_style_meta.pt")

    pd.DataFrame(metrics).to_csv(out / "results.csv", index=False)
    scores.rename("abs_pbcc").to_csv(out / "pbcc_scores.csv", header=True)
    if group_gate_summary is not None:
        pd.DataFrame([{"group": k, "mean_gate_weight": v} for k, v in group_gate_summary.items()]).to_csv(
            out / "group_gate_weights.csv", index=False
        )
    with (out / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "device": str(device), "epochs": args.epochs,
                   "batch_size": args.batch_size, "learning_rate": args.lr,
                   "resampling": args.resampling, "features": features,
                   "train_size_before_resampling": len(splits.y_train), "train_size_after_resampling": len(y_fit),
                   "validation_size": len(y_valid), "test_size": len(y_test),
                   "test_prevalence": float(y_test.mean()),
                   "mean_feature_group_gate_weights_on_test": group_gate_summary,
                   "feature_groups": {"clinical_history": ["HighBP", "HighChol", "CholCheck", "Stroke", "Diabetes", "DiffWalk"],
                                      "lifestyle": ["BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
                                      "self_reported_health": ["GenHlth", "MentHlth", "PhysHlth"],
                                      "access_demographics": ["AnyHealthcare", "NoDocbcCost", "Sex", "Age", "Education", "Income"]},
                   "note": "Structural reimplementation, not bit-for-bit reproduction; training-only resampling; threshold picked from a validation subset."}, f, indent=2)
    print("\nFinal test results:")
    print(pd.DataFrame(metrics)[["model", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc_average_precision"]].to_string(index=False))
    print(f"\nSaved artifacts to {out.resolve()}")


if __name__ == "__main__":
    main()
