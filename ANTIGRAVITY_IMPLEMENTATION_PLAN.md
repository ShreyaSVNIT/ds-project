# Antigravity Implementation Plan
## Reproduction and extension of EnsCVDD-Net / BlCVDD-Net

**Use this file as the agent's execution specification.** The workspace includes a starter implementation; audit and improve it rather than rewriting everything without inspection. The user has the real CSV and will place it in, or point Antigravity to, this workspace.

## 0. Project objective

Build three defensible deliverables from the paper *Heart Disease Prediction Using Novel Ensemble and Blending Based Cardiovascular Disease Detection Networks: EnsCVDD-Net and BlCVDD-Net* (IEEE Access, 2024; DOI `10.1109/ACCESS.2024.3421241`):

1. A reproducible implementation of the paper's main model family, to the extent the article describes it.
2. A research extension tested under a leakage-safe evaluation protocol.
3. Traceable outputs for the report and presentation: data audit, configs, model files, result CSVs, plots, ablations, and reproducibility notes.

**Do not promise any metric improvement.** The initial hypothesis is that structure-aware models and threshold selection can improve useful minority-class performance or robustness. Treat a +2 percentage-point accuracy improvement as a stretch target to test, not a requirement to manufacture.

## 1. First action: inspect before coding

Before editing files:

- Read `AGENTS.md`, `README.md`, `PROJECT_PLAN.md`, `theory_notes.md`, `requirements.txt`, all modules under `src/cardio/`, all scripts, and tests.
- Inspect `git status` and do not overwrite user work.
- Find the user's real CSV. First check `data/raw/`, workspace root, and any user-specified path. Identify it by the expected target column and actual header, not by filename alone. Ignore `synthetic_smoke.csv` and test fixtures unless explicitly running tests.
- Report what currently exists, what is incomplete, and any methodological/code bugs before changing it.
- Create a concise checklist in `IMPLEMENTATION_STATUS.md` with phases, commands, statuses (`TODO` / `IN PROGRESS` / `DONE` / `BLOCKED`), and evidence. Keep it current.

Use Antigravity's planning workflow (`/plan`) to produce a reviewable plan first. Do not start large experiments until the user approves the plan and the data audit has passed.

## 2. Exact dataset validation

Expected paper subset:
- Target: `HeartDiseaseorAttack`.
- 21 predictor columns, 22 columns total; paper describes 253,680 observations.
- Original class prevalence is approximately 9.4% positive. Recalculate from the actual CSV and use actual values in all reports.

Expected predictor names (the standard Kaggle-derived subset):
`HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `Diabetes`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`.

Implement a data audit script that outputs `artifacts/data_audit/` files:
- `dataset_audit.json`: path, SHA-256, byte size, dimensions, exact columns, dtypes, missing counts, duplicate-row count, target values/counts/prevalence, simple range/value checks, and software versions.
- `target_distribution.csv` and an EDA plot showing target balance.
- `feature_summary.csv` with unique counts, ranges and missingness per predictor.

If target or schema differs, do not guess, rename, or drop variables automatically. Print the detected header and request/record an explicit mapping. If the expected CSV is not found, ask for its path; do not train on synthetic fixtures.

## 3. Research question and pre-declared outcomes

Primary research question:

> Under a leakage-safe evaluation on the naturally imbalanced BRFSS-derived dataset, does a feature-group gated model improve minority-class ranking and/or operating-point performance relative to the paper-family architectures and strong tabular baselines?

Primary metric: **PR-AUC / average precision on the untouched test partition**.

Pre-declared operational metric: recall at a validation-selected threshold targeting at least 0.80 recall, with achieved test precision, specificity and F1 reported alongside it.

Secondary metrics: accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, Brier score, confusion matrix counts and fit/inference runtime.

The +2 percentage-point accuracy gain is a secondary stretch target. Do not change the primary metric after seeing test results. If no improvement is found, report the negative or mixed result honestly and interpret its limitations.

Before writing a novelty claim, perform a short literature check for feature-group/feature-gated tabular neural models on health survey or cardiovascular data. Save source links and a 5–10 row novelty matrix in `research/related_work_matrix.md`. If the environment cannot access reliable external papers, state that and do not claim the architecture is the first of its kind.

## 4. Experimental design — must be leakage-safe

### Data splits
- Create a fixed, stratified 70% train / 15% validation / 15% test split with seed 42 for the initial development run.
- Save row indices or stable row identifiers for each partition to enable exact reuse; never include the target in predictors.
- Verify and log target prevalence separately in train, validation and test. They should remain close to the source distribution.
- Keep the test split untouched until the model and threshold policy are frozen.

### Preprocessing order
1. Load raw data and validate schema.
2. Split raw rows.
3. Fit scaler/imputer/feature ranking on training only where needed.
4. Apply the fitted transform unchanged to validation/test.
5. Apply resampling only to the training partition. For cross-validation or OOF predictions, resampling must occur *inside each fold's training data* via a compatible `imblearn.pipeline.Pipeline`.
6. Train, choose settings/threshold on validation, freeze settings, then evaluate test.

Important existing-code audit: `run_tabular_experiments.py` currently resamples the full training partition before `StackingClassifier` creates its internal cross-validation folds when `--resampling adasyn` is used. That can leak information through synthetic samples across OOF folds. Refactor so each fold has its own train-fold-only preprocessing/resampling, or disable resampling for that stack path and explain the alternative. Add a regression test.

Do not describe `top_k=21` as effective feature selection; it keeps all 21 features. Evaluate all-feature models and PBCC top-k variants explicitly. Any choice of `k` must be made from validation results only. PBCC scores/rankings must use training labels only.

### Imbalance strategies
Compare separately:
- No resampling (primary, naturally sampled train data).
- Class-weighted loss/class weights.
- ADASYN applied to training only.
- Optional random oversampling as a control.

ADASYN may produce fractional synthetic values in binary or ordinal survey variables. Do not silently round them. Describe the issue and keep ADASYN as an experimental ablation, not an unquestioned default.

## 5. Model matrix

Implement, test and compare in phases. Do not run the full matrix before the small validation run works.

### Phase A — cheap baselines
- Majority-class `DummyClassifier` and prevalence baseline.
- Logistic Regression with class weights and scaling.
- Random Forest with balanced class weights.
- Extra Trees with balanced class weights.
- Histogram Gradient Boosting (with a documented imbalance strategy).
- Diverse stacked tabular blender; generate OOF base predictions correctly.

### Phase B — paper-family neural models
- LeNet1D-like model.
- GRU model.
- EnsCVDD-style parallel LeNet+GRU feature hybrid.
- BlCVDD-style stacking/blending with an MLP meta-learner.

Document exact differences from paper because the article has descriptions/tables but no executable source code. Do not call the implementation an exact reproduction unless architecture, preprocessing, split and training details genuinely match.

### Phase C — proposed extension: Feature-Group Gated Network (FGGN)
Use the four domain groups below; validate every feature name against the actual dataset before indexing.
- `clinical_history`: `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `Diabetes`, `DiffWalk`.
- `lifestyle`: `BMI`, `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`.
- `self_reported_health`: `GenHlth`, `MentHlth`, `PhysHlth`.
- `access_demographics`: `AnyHealthcare`, `NoDocbcCost`, `Sex`, `Age`, `Education`, `Income`.

Each group gets a small MLP encoder: `h_g = f_g(x_g)`. A gate uses all group embeddings to produce `alpha = softmax(W[h_1;...;h_G] + b)`. Fuse embeddings using `h = sum_g alpha_g h_g`, then predict the target using a binary classifier with logits and `BCEWithLogitsLoss` or a properly justified class-weighted alternative. Ensure all group embeddings have the same dimension. Save per-row or mean gate weights on validation, and never interpret weights causally.

The architecture is an empirical proposed extension; do not label it novel/first until the related-work matrix supports the claim.

### Phase D — ablations and robustness
At minimum compare:
1. FGGN with gates versus the same group encoders with uniform averaging/no learned gate.
2. All 21 features versus training-only PBCC top-k selection.
3. No resampling versus class weights versus ADASYN.
4. Fixed threshold 0.50 versus validation-selected threshold for target recall 0.80.
5. Single seed baseline versus at least 3 seeds for the selected models. Prefer stratified 5-fold CV if time/computing resources permit; if using repeated folds, keep preprocessing and resampling fold-local.

## 6. Threshold policy

- Always report fixed 0.50 metrics.
- Also calculate a threshold on validation data only: among thresholds achieving recall >= 0.80 on validation, choose the threshold with the highest precision (with deterministic tie-breaking).
- Store chosen threshold and validation recall/precision in run metadata.
- Evaluate that frozen threshold on test and report achieved test recall, precision, specificity and confusion matrix; do not claim the validation recall target is guaranteed on test.
- If no threshold meets the constraint, report that explicitly rather than hiding it.

## 7. Code quality and tests

Audit/refactor the existing scaffold. Keep scripts small and reusable. Add a configuration file or explicit CLI options for seed, path, model, resampling, `top_k`, epochs, batch size, threshold target and output directory. Record package versions.

Required automated tests:
- Target/schema contract and informative failure on wrong CSV.
- Train/validation/test partition disjointness and stratified prevalence.
- PBCC feature selection fits from training data only.
- Resampling never touches validation/test.
- OOF stacking folds do not share ADASYN-generated samples across train/holdout folds.
- Threshold satisfies validation recall target when feasible.
- Metrics are computed from probabilities and a frozen threshold.
- FGGN outputs one logit per row, gate weights sum to 1 and all declared feature groups are handled.
- Saved results/metadata include model, seed, split sizes, prevalence, threshold, primary metrics and output paths.

Run `pytest -q` and a `--quick`/few-epoch real-data end-to-end test before longer training. If dependencies or hardware limitations prevent a test, report the exact command and error; do not say it passed.

## 8. Output structure

Maintain/add this structure without committing the CSV or large model files unless asked:

```text
AGENTS.md
ANTIGRAVITY_IMPLEMENTATION_PLAN.md
IMPLEMENTATION_STATUS.md
research/
  related_work_matrix.md
  paper_reproduction_notes.md
configs/
  baseline.yaml              # or equivalent documented config
src/cardio/
  data.py
  metrics.py
  preprocessing.py
  tabular_models.py
  torch_models.py
  evaluation.py
scripts/
  audit_dataset.py
  run_tabular_experiments.py
  run_paper_models.py
  run_ablation_study.py
  explain_shap.py
  make_report_figures.py
artifacts/
  data_audit/
  tabular/<run_id>/
  neural/<run_id>/
  figures/
  final_results/
tests/
```

For every run, save `results.csv`, `run_metadata.json`, a readable config, model weights where appropriate, confusion-matrix plot, PR curve and ROC curve. Add `SHA256SUMS.txt` for final deliverables. Do not put fabricated metrics in templates.

## 9. Report and presentation outputs

Once experiments genuinely finish, generate draft assets for:
- `report/outline.md`: abstract placeholder, introduction, related work, problem formulation, dataset, original models, proposed method/equations, experimental protocol, results, ablations, interpretability, threats to validity and conclusion.
- `presentation/slide_outline.md`: 10–12 slide sequence: problem, paper summary, research gap, contribution, dataset and prevalence, method diagram, implementation, evaluation protocol, results, ablation/SHAP, limitations and conclusion.
- Tables/plots from actual `results.csv`, never manually typed result claims.

Clearly distinguish values quoted from the original paper, values reproduced by our implementation, and values from the proposed method. Cite the paper and any literature references. Do not make clinical recommendations or causal claims.

## 10. Acceptance criteria

The minimum version is considered complete only when:
- The actual CSV is validated and the audit report is saved.
- Baselines and at least one paper-family neural architecture run successfully on the real data.
- FGGN trains and saves a checkpoint; unit tests pass.
- Resampling is training-fold-only and test prevalence is preserved.
- Threshold tuning uses validation only.
- Model comparison includes PR-AUC, recall, precision, specificity, F1, accuracy, ROC-AUC, Brier score, confusion counts and runtime.
- A run can be reproduced from documented commands/config and seeds.
- No result or novelty statement is claimed without evidence.

## 11. Required progress updates from Antigravity

At the end of every phase, report:
1. Files inspected/changed.
2. Commands actually run and whether they passed.
3. Real dataset shape and class prevalence (after data audit only).
4. Metrics observed (after a real run only).
5. Any deviations from the plan and unresolved risks.
6. The next single recommended step.

Do not silently skip phases or launch expensive runs. Stop and ask if the real CSV schema does not match, if an important methodology choice cannot be resolved from the paper, or if the next step would overwrite data/results.
