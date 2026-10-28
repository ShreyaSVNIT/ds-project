import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.data import (
    TARGET,
    DataSplits,
    load_dataset,
    load_split_indices,
    make_splits,
    pbcc_scores,
    save_split_indices,
    select_top_pbcc,
)
from cardio.metrics import choose_threshold_for_recall, metric_dict


def test_split_preserves_prevalence_and_disjointness():
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    # 9.4% prevalence
    pos = int(n * 0.094)
    y_arr = np.zeros(n, dtype=int)
    y_arr[:pos] = 1
    rng.shuffle(y_arr)
    y = pd.Series(y_arr)

    splits = make_splits(X, y, seed=42, test_size=0.15, valid_size=0.15)
    # Check prevalence preservation
    assert abs(splits.y_train.mean() - y.mean()) < 0.015
    assert abs(splits.y_valid.mean() - y.mean()) < 0.02
    assert abs(splits.y_test.mean() - y.mean()) < 0.02

    # Check partition disjointness
    tr_idx = set(splits.train_indices)
    va_idx = set(splits.valid_indices)
    te_idx = set(splits.test_indices)
    assert len(tr_idx & va_idx) == 0
    assert len(tr_idx & te_idx) == 0
    assert len(va_idx & te_idx) == 0
    assert len(tr_idx | va_idx | te_idx) == n


def test_save_and_load_split_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "indices.json"
        splits = DataSplits(
            X_train=pd.DataFrame({"a": [1]}),
            X_valid=pd.DataFrame({"a": [2]}),
            X_test=pd.DataFrame({"a": [3]}),
            y_train=pd.Series([0]),
            y_valid=pd.Series([1]),
            y_test=pd.Series([0]),
            train_indices=np.array([0, 1, 2]),
            valid_indices=np.array([3]),
            test_indices=np.array([4, 5]),
        )
        save_split_indices(splits, tmp_path)
        loaded = load_split_indices(tmp_path)
        np.testing.assert_array_equal(loaded["train_indices"], np.array([0, 1, 2]))
        np.testing.assert_array_equal(loaded["valid_indices"], np.array([3]))
        np.testing.assert_array_equal(loaded["test_indices"], np.array([4, 5]))


def test_schema_contract_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "bad.csv"
        # Missing target
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Expected target column"):
            load_dataset(csv_path)

        # Non-binary target
        pd.DataFrame({TARGET: [0, 2], "a": [1, 2]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Target must be binary 0/1"):
            load_dataset(csv_path)

        # Missing values in target
        pd.DataFrame({TARGET: [0, np.nan], "a": [1, 2]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Target contains missing values"):
            load_dataset(csv_path)


def test_pbcc_feature_selection_fits_from_train_only():
    # 'strong' correlates with ytr, 'leak_feature' correlates only in validation
    Xtr = pd.DataFrame({"strong": [0, 0, 1, 1, 0, 1], "weak": [1, 0, 1, 0, 1, 0], "leak_feature": [0, 0, 0, 0, 0, 0]})
    ytr = pd.Series([0, 0, 1, 1, 0, 1])
    Xv = Xtr.copy()
    Xte = Xtr.copy()
    A, B, C, features, scores = select_top_pbcc(Xtr, Xv, Xte, ytr, 1)
    assert features == ["strong"]
    assert list(A.columns) == list(B.columns) == list(C.columns) == ["strong"]


def test_threshold_recall_constraint():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.55, 0.7, 0.9])
    threshold = choose_threshold_for_recall(y, p, minimum_recall=2 / 3)
    assert np.mean((p[y == 1] >= threshold)) >= 2 / 3
    metrics = metric_dict(y, p, threshold)
    assert "pr_auc_average_precision" in metrics
    assert "brier_score" in metrics
    assert "specificity" in metrics
