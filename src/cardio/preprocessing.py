from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler
from sklearn.preprocessing import StandardScaler


def resample_train(X: pd.DataFrame, y: pd.Series, method: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Resample training partition only.
    Standardizes for neighbor distances during ADASYN, then inverse-transforms
    so all downstream models receive consistent scales.
    """
    if method == "none":
        return X.copy(), y.copy()
    if method == "adasyn":
        sampler = ADASYN(random_state=seed, n_neighbors=5)
    elif method == "random":
        sampler = RandomOverSampler(random_state=seed)
    else:
        raise ValueError(f"Unknown resampling strategy: {method!r}. Must be 'none', 'adasyn', or 'random'.")

    distance_scaler = StandardScaler().fit(X)
    X_scaled = distance_scaler.transform(X)
    X_res_scaled, y_res = sampler.fit_resample(X_scaled, y)
    X_res = distance_scaler.inverse_transform(X_res_scaled)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
