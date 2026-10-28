import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.metrics import choose_threshold_for_recall, metric_dict
from cardio.tabular_models import LeakageSafeStackedBlender, build_tabular_models


def test_leakage_safe_stacked_blender():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(rng.integers(0, 2, size=120))

    base_lr = LogisticRegression()
    blender = LeakageSafeStackedBlender(
        base_estimators=[("lr1", base_lr)],
        final_estimator=LogisticRegression(),
        cv=3,
        resampling="none",
        seed=42,
    )
    blender.fit(X, y)
    probs = blender.predict_proba(X)
    assert probs.shape == (120, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds = blender.predict(X)
    assert len(preds) == 120
    assert set(preds).issubset({0, 1})


def test_build_tabular_models():
    models = build_tabular_models(seed=42, quick=True)
    expected = [
        "majority_dummy",
        "logistic_regression_balanced",
        "random_forest_balanced",
        "extra_trees_balanced",
        "hist_gradient_boosting",
        "lrcb_stacked_blender",
    ]
    for name in expected:
        assert name in models, f"Model {name} missing from build_tabular_models"


def test_vectorized_threshold_selection_matches_constraint():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.array([0.05, 0.1, 0.2, 0.35, 0.4, 0.45, 0.6, 0.7, 0.85, 0.9])
    # Target recall 0.80 -> requires at least 4 of 5 positive cases detected
    thresh = choose_threshold_for_recall(y, probs, minimum_recall=0.80)
    recalled = np.mean(probs[y == 1] >= thresh)
    assert recalled >= 0.80
    metrics = metric_dict(y, probs, threshold=thresh)
    assert metrics["recall_sensitivity"] >= 0.80
    assert metrics["precision"] > 0.0
