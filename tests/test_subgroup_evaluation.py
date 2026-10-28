from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_subgroups import evaluate_slice, map_demographics, wilson_interval



def test_wilson_interval():
    # 0 total
    low, high = wilson_interval(0, 0)
    assert low == 0.0 and high == 0.0

    # 50 out of 100
    low, high = wilson_interval(50, 100)
    assert 0.39 < low < 0.42
    assert 0.58 < high < 0.61

    # 0 out of 100
    low, high = wilson_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 0.05


def test_map_demographics():
    df = pd.DataFrame({
        "Sex": [0.0, 1.0, 0.0],
        "Age": [1.0, 7.0, 12.0],
        "Income": [2.0, 5.0, 8.0],
    })
    mapped = map_demographics(df)
    assert mapped["Sex_label"].tolist() == ["Female", "Male", "Female"]
    assert mapped["Age_bracket"].tolist() == ["Young Adults (18-44)", "Middle-Aged (45-64)", "Older Adults (65+)"]
    assert mapped["Income_tier"].tolist() == ["Lower Income (<$25k)", "Middle Income ($25k-<$50k)", "Higher Income (>=$50k)"]


def test_evaluate_slice_metrics():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # TP=2, FP=1, FN=0, TN=7

    res = evaluate_slice(y_true, y_prob, y_pred)
    assert res["n_obs"] == 10
    assert res["n_positives"] == 2
    assert res["prevalence"] == 0.2
    assert res["unreliable_sample"] is True  # N < 100
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["tn"] == 7
    assert res["recall"] == 1.0
    assert abs(res["precision"] - 2.0 / 3.0) < 1e-4
    assert res["specificity"] == 0.875
    assert 0.8 < res["pr_auc"] <= 1.0
    assert 0.9 < res["roc_auc"] <= 1.0
