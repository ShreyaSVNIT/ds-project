# Workspace Rules — Cardiovascular Classification Research Project

## Purpose
Build a reproducible research implementation of the 2024 IEEE Access paper on EnsCVDD-Net and BlCVDD-Net, then evaluate a clearly documented extension. This is a course research prototype, not a clinical tool.

## Non-negotiable research rules
1. Do not invent experiment results, dataset facts, citations, or completed tasks. Label expected results as hypotheses until measured.
2. Do not claim a 2-percentage-point accuracy improvement unless computed on a held-out, untouched test set and reported alongside the baseline on the same split. Treat 2 percentage points as a stretch target, not a guarantee.
3. Preserve the naturally imbalanced validation and test partitions. Never resample, augment, fit scalers, select features, impute, calibrate, or choose thresholds using the test set.
4. Split raw rows first. Fit all learned preprocessing and PBCC feature ranking using training data only. Apply ADASYN only to a training fold or training partition, never before cross-validation splitting.
5. If stacking/blending uses out-of-fold (OOF) predictions, every fold's preprocessing/resampling must be fitted only on that fold's training portion. Never globally oversample before generating OOF predictions.
6. Choose hyperparameters, final models, and operating thresholds using training/validation data. Freeze the decision before final test evaluation. Record thresholds and test class prevalence.
7. Accuracy is not sufficient for a roughly 9.4% positive class. Always include PR-AUC (average precision), recall/sensitivity, precision, specificity, F1, balanced accuracy, ROC-AUC, Brier score, confusion counts, runtime, and prevalence.
8. ADASYN interpolates encoded survey features and can create semantically invalid fractional values for binary/ordinal features. Treat it as an ablation, compare against class weighting and no resampling, and document the limitation.
9. Feature-group gate weights and SHAP values are model explanations, not causal effects. Never claim clinical readiness or medical advice.
10. The original paper does not provide executable source code in the supplied article. Call our models structural reimplementations unless exact equivalence is independently verified.

## Dataset contract
- Expected target: `HeartDiseaseorAttack`.
- Expected Kaggle-derived paper subset: 253,680 rows, 21 predictors plus the target (22 columns), subject to checking the actual supplied CSV.
- Detect and validate the user's real CSV. Prefer an explicit `--data` path; if absent, inspect likely locations and list candidate files. Do not accidentally train on `synthetic_smoke.csv` or other toy test fixtures.
- Verify target values are binary 0/1, predictors are numeric/encoded as expected, there are no duplicate column names, and missingness/duplicate rows/ranges are reported. If the schema is different, stop and explain the mismatch rather than silently guessing the target or dropping columns.
- Do not commit or copy a large dataset into Git unless the user explicitly requests it and redistribution terms allow it. Keep paths configurable.

## Engineering standards
- Inspect the existing code before editing; preserve working code and make small, auditable changes.
- Use typed, modular Python functions, clear error messages, fixed random seeds, and explicit configuration/CLI arguments.
- Add tests for schema validation, split prevalence, preprocessing boundaries, OOF stacking leakage protection, threshold selection, and saved metric outputs.
- Save model/config metadata, dataset fingerprint (e.g. SHA-256), seed, package versions, class prevalence, chosen features/threshold, training time, and output paths with every run.
- Never overwrite prior results silently; use distinct run directories or ask before destructive changes.
- Keep smoke tests small, but label synthetic data clearly and never present it as research evidence.
- After each phase, run relevant tests and show the actual command/output summary, unresolved issues, and next step.

## Required working behaviour
- For a substantial change, first create/review an implementation plan and wait for user approval before broad code modifications.
- Work in phases: inspect → validate data → baselines → paper-family models → proposed model → ablations/robustness → final artifacts/report.
- Do not start long runs until a small end-to-end run passes.
- Prefer reproducibility and valid comparisons over trying many arbitrary models to chase a score.
