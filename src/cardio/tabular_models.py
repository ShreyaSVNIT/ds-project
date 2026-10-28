from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardio.preprocessing import resample_train


class LeakageSafeStackedBlender(BaseEstimator, ClassifierMixin):
    """Stacked blender that guarantees complete isolation of resampling across out-of-fold splits.
    
    During cross-validation for meta-feature generation, resampling (e.g. ADASYN) is performed
    strictly on the training sub-fold. The validation sub-fold remains completely untouched and
    retains natural prevalence.
    """
    def __init__(
        self,
        base_estimators: list[tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator | None = None,
        cv: int = 3,
        resampling: str = "none",
        seed: int = 42,
    ):
        self.base_estimators = base_estimators
        self.final_estimator = (
            final_estimator
            if final_estimator is not None
            else LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
        )
        self.cv = cv
        self.resampling = resampling
        self.seed = seed
        self.fitted_base_models: list[BaseEstimator] = []
        self.fitted_final_estimator: BaseEstimator | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)
        n_samples = len(X_df)
        n_models = len(self.base_estimators)
        oof_preds = np.zeros((n_samples, n_models), dtype=float)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for tr_idx, va_idx in skf.split(X_df, y_ser):
            X_tr_fold = X_df.iloc[tr_idx]
            y_tr_fold = y_ser.iloc[tr_idx]
            X_va_fold = X_df.iloc[va_idx]

            # Resample strictly on fold training data
            X_tr_res, y_tr_res = resample_train(X_tr_fold, y_tr_fold, self.resampling, self.seed)

            for m_idx, (name, estimator) in enumerate(self.base_estimators):
                cloned = clone(estimator)
                cloned.fit(X_tr_res, y_tr_res)
                # Predict probability on untouched validation fold
                oof_preds[va_idx, m_idx] = cloned.predict_proba(X_va_fold)[:, 1]

        # Fit final estimator on honest OOF predictions and natural training labels
        self.fitted_final_estimator = clone(self.final_estimator)
        self.fitted_final_estimator.fit(oof_preds, y_ser)

        # Fit final base models on the fully resampled training partition
        X_full_res, y_full_res = resample_train(X_df, y_ser, self.resampling, self.seed)
        self.fitted_base_models = []
        for name, estimator in self.base_estimators:
            full_model = clone(estimator)
            full_model.fit(X_full_res, y_full_res)
            self.fitted_base_models.append(full_model)

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.fitted_final_estimator is None or not self.fitted_base_models:
            raise RuntimeError("LeakageSafeStackedBlender must be fitted before predict_proba.")
        base_probs = []
        for model in self.fitted_base_models:
            base_probs.append(model.predict_proba(X)[:, 1])
        meta_features = np.column_stack(base_probs)
        return self.fitted_final_estimator.predict_proba(meta_features)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def build_tabular_models(seed: int, resampling: str = "none", quick: bool = False) -> dict[str, BaseEstimator]:
    trees = 80 if quick else 160
    base_lr = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=300, solver="lbfgs", random_state=seed)),
    ])
    rf = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed, max_features="sqrt"
    )
    et = ExtraTreesClassifier(
        n_estimators=trees, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=seed, max_features="sqrt"
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=80 if quick else 150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed
    )
    blender = LeakageSafeStackedBlender(
        base_estimators=[("lr", base_lr), ("et", et), ("rf", rf)],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed),
        cv=3,
        resampling=resampling,
        seed=seed,
    )
    return {
        "majority_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression_balanced": base_lr,
        "random_forest_balanced": rf,
        "extra_trees_balanced": et,
        "hist_gradient_boosting": hgb,
        "lrcb_stacked_blender": blender,
    }
