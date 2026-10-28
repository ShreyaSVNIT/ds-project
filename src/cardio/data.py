from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "HeartDiseaseorAttack"
EXPECTED_ROWS_APPROX = 253_680


import json

@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray | None = None
    valid_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None


def save_split_indices(splits: DataSplits, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_indices": splits.train_indices.tolist() if splits.train_indices is not None else [],
        "valid_indices": splits.valid_indices.tolist() if splits.valid_indices is not None else [],
        "test_indices": splits.test_indices.tolist() if splits.test_indices is not None else [],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: np.asarray(v, dtype=int) for k, v in data.items()}


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` "
            "or download the Kaggle CSV and place it there."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column {TARGET!r}. The paper's 22-column Kaggle subset is "
            "required; the full CDC BRFSS file uses a different schema."
        )
    if df.columns.duplicated().any():
        raise ValueError("Dataset has duplicated column names.")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values; document and handle them before training.")
    y = df.pop(TARGET).astype(int)
    X = df.apply(pd.to_numeric, errors="raise")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found {sorted(y.unique().tolist())}.")
    if X.isna().any().any():
        raise ValueError("Predictors contain missing values; add a documented imputation strategy first.")
    if len(X.columns) != 21:
        print(f"WARNING: expected 21 predictors, found {len(X.columns)}. Check the dataset version.")
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    test_size: float = 0.15,
    valid_size: float = 0.15,
) -> DataSplits:
    """Stratified train/validation/test split; validation and test keep original prevalence."""
    if test_size <= 0 or valid_size <= 0 or test_size + valid_size >= 1:
        raise ValueError("test_size and valid_size must be > 0 and sum to < 1.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + valid_size, random_state=seed, stratify=y
    )
    relative_test = test_size / (test_size + valid_size)
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_temp, y_temp, test_size=1 - relative_test, random_state=seed, stratify=y_temp
    )
    train_idx = np.asarray(X_train.index, dtype=int)
    valid_idx = np.asarray(X_valid.index, dtype=int)
    test_idx = np.asarray(X_test.index, dtype=int)
    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        train_indices=train_idx,
        valid_indices=valid_idx,
        test_indices=test_idx,
    )


def pbcc_scores(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """Absolute point-biserial scores, equivalent to Pearson correlation with a binary target."""
    y_arr = np.asarray(y_train, dtype=float)
    scores: dict[str, float] = {}
    for col in X_train.columns:
        x = np.asarray(X_train[col], dtype=float)
        if np.std(x) == 0 or np.std(y_arr) == 0:
            scores[col] = 0.0
        else:
            scores[col] = float(abs(np.corrcoef(x, y_arr)[0, 1]))
    return pd.Series(scores).sort_values(ascending=False)


def select_top_pbcc(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    top_k: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    scores = pbcc_scores(X_train, y_train)
    if top_k is None or top_k >= len(scores):
        selected = list(X_train.columns)
    elif top_k < 1:
        raise ValueError("top_k must be None or a positive integer.")
    else:
        selected = scores.head(top_k).index.tolist()
    return X_train[selected].copy(), X_valid[selected].copy(), X_test[selected].copy(), selected, scores
