#!/usr/bin/env python3
"""Generate a small, clearly synthetic dataset for pipeline smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/synthetic_smoke.csv")
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.rows < 60:
        raise ValueError("Synthetic smoke data requires at least 60 rows.")

    rng = np.random.default_rng(args.seed)
    features = rng.normal(size=(args.rows, 21))
    signal = 1.2 * features[:, 0] - 0.8 * features[:, 1] + 0.5 * features[:, 2]
    probability = 1 / (1 + np.exp(-signal))
    target = (rng.random(args.rows) < probability).astype(int)

    # Ensure both classes remain available after stratified splitting.
    target[0] = 0
    target[1] = 1
    frame = pd.DataFrame(features, columns=[f"feature_{i:02d}" for i in range(1, 22)])
    frame.insert(0, "HeartDiseaseorAttack", target)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote synthetic smoke dataset with {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
