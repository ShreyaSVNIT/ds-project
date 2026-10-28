from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metric_dict(y_true, prob, threshold: float = 0.5) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return result


from sklearn.metrics import precision_recall_curve


def choose_threshold_for_recall(y_true, probabilities, minimum_recall: float = 0.80) -> float:
    """Return threshold maximizing precision on validation data subject to recall >= minimum_recall.
    Tie-breaks deterministically by highest threshold.
    """
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1].")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y, p)
    # precision_recall_curve returns len(thresholds) + 1 elements for precision and recall
    p_cand = precisions[:-1]
    r_cand = recalls[:-1]
    t_cand = thresholds

    # Include boundary threshold 0.0 where all instances are predicted positive
    p_0 = float(np.mean(y)) if len(y) > 0 else 0.0
    r_0 = 1.0
    t_0 = 0.0

    all_p = np.concatenate(([p_0], p_cand))
    all_r = np.concatenate(([r_0], r_cand))
    all_t = np.concatenate(([t_0], t_cand))

    mask = all_r >= minimum_recall
    if not np.any(mask):
        return 0.0

    valid_p = all_p[mask]
    valid_t = all_t[mask]

    # lexsort((secondary, primary)) sorts by primary first, then secondary
    best_idx = np.lexsort((valid_t, valid_p))[-1]
    return float(valid_t[best_idx])
