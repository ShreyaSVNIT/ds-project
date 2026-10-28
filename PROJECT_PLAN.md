# Primary Project Plan

## Working title
**Leakage-Aware, Recall-Constrained Blending for Imbalanced Cardiovascular Risk Classification: A Reproduction and Audit of EnsCVDD-Net / BlCVDD-Net**

## 1. Why this paper and the opportunity
The reference paper (IEEE Access, 2024) combines LeNet1D and GRU in EnsCVDD-Net and blends their predictions with an MLP in BlCVDD-Net. It uses ADASYN for imbalance, point-biserial correlation for feature ranking/selection, 10-fold cross-validation and SHAP. The reported highest BlCVDD-Net result is 91.7% accuracy and 91.2% F1 at 30 epochs.

Our goal is not to claim novelty just by changing one classifier. We will reproduce the central idea as far as available detail allows, then test a principled extension that addresses the evaluation risks that matter for an imbalanced tabular health dataset.

## 2. Proposed contribution
**Feature-Group Gated Network (FGGN) + leakage-aware, recall-constrained evaluation.**
- Partition the 21 inputs into four semantically motivated groups: clinical history; lifestyle; self-reported health; healthcare access/demographics. Give each group its own small MLP encoder.
- Learn a softmax gate over the group embeddings and fuse them before the final binary classifier. The gate provides a coarse group-level attribution in addition to feature-level SHAP on a tree baseline.
- Train only on the training partition; fit PBCC selection and any scaling on training data only.
- Apply ADASYN only to training data. Keep validation and test rows untouched, with the original prevalence.
- For the neural BlCVDD-style meta-learner, fit on one validation subset and select its operating threshold on another validation subset. For the tabular blender use out-of-fold predictions for the meta-learner.
- Use validation data to pick the highest classification threshold that still meets a declared recall target (default 0.80), maximizing precision subject to that recall constraint.
- Evaluate both standard threshold 0.50 and tuned threshold. Include class-sensitive metrics and Brier score/precision-recall analysis.
- Compare FGGN with paper-family CNN/GRU, standard tabular learners, and imbalance-ablation variants.

The contribution should be phrased as an *empirical hypothesis*. Do not write “we improve accuracy by 2%” until the result is measured on the untouched test set. Define “2%” precisely as either 2 percentage points (e.g. 91.0% to 93.0%) or a relative 2% change.


## 2A. Paper audit finding to motivate the extension
The source dataset has 23,893 positive cases out of 253,680 (about 9.4%), so a majority-class predictor alone is about 90.6% accurate on the original distribution. In the paper's 30-epoch Table 14 (printed page 109249 / PDF page 20), the reported BlCVDD-Net values are approximately 91.7% accuracy, 96.0% precision and 87.8% recall; base GRU metrics are also near 90.5% across accuracy, precision, recall and F1. Those metric patterns are consistent with a near-balanced evaluation set, not the natural 9.4% positive prevalence. The article's flow diagram and description should be audited to establish exactly when ADASYN is applied and what prevalence is used for each test fold. Do **not** claim leakage is proven from the paper alone; treat the unclear evaluation prevalence / resampling order as a reproducibility concern. Our main fix is to report held-out results on untouched, naturally imbalanced rows.

## 3. Dataset and split
- Paper dataset: `Heart Disease Health Indicators Dataset`, derived from CDC BRFSS 2015; 253,680 rows and 22 columns; target `HeartDiseaseorAttack`.
- Start with a stratified 70/15/15 split into train / validation / test, fixed seed.
- Preserve original class prevalence in all three partitions; do not resample validation/test.
- For a final robustness table, repeat with stratified 5-fold cross-validation or at least 3 seeds. If compute is limited, first establish the single split, then add repeats.
- Optional generalization check: use the official BRFSS documentation / another compatible year only after reconciling the schema and outcome definition. This is not part of the minimum viable implementation.

## 4. Experiment matrix

| ID | Model / ablation | Imbalance handling | Threshold | Purpose |
|---|---|---|---|---|
| E0 | Majority-class dummy | None | N/A | Sanity baseline |
| E1 | Logistic Regression | Class weights | 0.50 and validation-tuned | Transparent baseline |
| E2 | Random Forest | Balanced class weights | 0.50 and validation-tuned | Nonlinear tree baseline |
| E3 | Extra Trees | Balanced class weights | 0.50 and validation-tuned | Strong tabular baseline |
| E4 | HistGradientBoosting | None / optional sample weights | 0.50 and validation-tuned | Gradient-boosted tree baseline |
| E5 | Diverse stacked blender | Base-level class weights; OOF meta features | 0.50 and recall-constrained | Main proposed tabular extension |
| E6 | LeNet1D | ADASYN training-only, then class-weight ablation | 0.50 | Paper-family single model |
| E7 | GRU | ADASYN training-only, then class-weight ablation | 0.50 | Paper-family single model |
| E8 | EnsCVDD-style LeNet+GRU | ADASYN training-only | 0.50 | Paper-family hybrid |
| E9 | BlCVDD-style LeNet+GRU+MLP | ADASYN training-only; validation predictions train meta-model | 0.50 | Paper-family blending |
| E10 | Feature-Group Gated Network (FGGN) | Class weighting or training-only ADASYN | 0.50 and validation-tuned | Proposed group-aware tabular architecture |
| E11 | Ablation: remove PBCC, remove ADASYN, disable threshold tuning / remove group gate | One factor changed at a time | As applicable | Determine which component helps |

Do not run every row at once. M0 first: E0–E5. M1: E6–E9 if compute/time permits. M2: repeat best models across seeds/folds and add SHAP/subgroup analyses.

## 5. Metrics
Primary: PR-AUC and recall at a predeclared operating point. Secondary: F1, precision, specificity, balanced accuracy, ROC-AUC, accuracy, Brier score, runtime, and confidence intervals across repeated folds/seeds. Show confusion matrices with each result table. Report metrics on the original imbalanced test set.

## 6. Work division (suggested for 4 people)
- **Member A — paper & theory:** literature review, reproduce methodology diagram, equations for PBCC/ADASYN/CNN/GRU/stacking, identify gaps and write related work.
- **Member B — data & validity:** dataset provenance, schema verification, EDA, split policy, imbalance/feature-selection ablations, reproducibility log.
- **Member C — models & evaluation:** tabular baselines, neural models, threshold selection, repeated runs, metrics and plots.
- **Member D — explainability & submission:** SHAP and subgroup checks, limitations, paper formatting, PPT/demo integration. All members should review the final methods and results.

## 7. Milestones
1. **Day 1:** download/check data; inspect target prevalence; reproduce stratified split and majority baseline.
2. **Days 2–3:** finish LR/RF/ExtraTrees/HistGradientBoosting baselines; create first results CSV.
3. **Days 4–5:** implement/test stacked blender and validation threshold selection; ablate ADASYN and PBCC.
4. **Days 6–8:** run neural paper-family models if feasible; otherwise scope them to a reproducible subset and explicitly label it.
5. **Days 9–10:** multiple seeds/folds, confidence intervals, SHAP; identify final best method.
6. **Final phase:** report, PPT, environment/dependency check, end-to-end reproducibility run.

## 8. Decision rules
- Success metric and target recall must be declared before final test evaluation.
- Never choose the final model or threshold using the test labels.
- A nominal 2 percentage-point gain only counts if test evaluation and repeated runs support it; include uncertainty and runtime.
- If accuracy cannot improve, report a meaningful trade-off such as recall/PR-AUC improvement at comparable specificity or reduced runtime. Do not hide the trade-off.
- Avoid causal claims in SHAP plots and avoid claims of clinical readiness.
