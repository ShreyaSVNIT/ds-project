# Research Paper Blueprint & Outline
# Title: Leakage-Aware Evaluation of Tabular Models for Classifying Self-Reported Heart Disease History in BRFSS 2015
## An Auditable Reimplementation and Multi-Seed Extension of EnsCVDD-Net

---

## Abstract
Recent literature has proposed hybrid deep neural architectures combining 1D Convolutional Neural Networks and Gated Recurrent Units (EnsCVDD-Net, BlCVDD-Net) for cardiovascular disease detection from tabular health survey data, reporting accuracies exceeding 91% and precisions near 96% with ADASYN oversampling. In this study, we conduct an auditable reimplementation and methodological audit of these architectures on the 253,680-row CDC BRFSS 2015 dataset. We demonstrate through mathematical derivation that the reported precision/recall metrics are mutually incompatible with the reported accuracy under the natural positive survey prevalence (9.42%), where a trivial majority-class classifier achieves 90.58% accuracy. We introduce a leakage-safe evaluation protocol that strictly confines preprocessing, PBCC feature ranking, and class weighting to training folds, preserving untouched validation and test distributions. Furthermore, we investigate the **Feature-Group Gated Network (FGGN)** and an un-gated **UniformGroupNet**, which replace arbitrary 1D column ordering with four physiologically grounded domain encoders (*Clinical History*, *Lifestyle*, *Self-Reported Health*, *Demographics/Access*). In a rigorous 5-seed study (Seeds 42, 101, 2024, 777, 9999), tabular Histogram Gradient Boosting achieves the highest overall ranking performance (PR-AUC: $0.3717 \pm 0.0067$, ROC-AUC: $0.8483 \pm 0.0026$). While both domain-grouped architectures outperform EnsCVDD-Net (PR-AUC: $0.3563 \pm 0.0055$), learned softmax gating ($0.3689 \pm 0.0068$) provides no performance advantage over uniform semantic group pooling ($0.3693 \pm 0.0070$), representing a key neutral finding on model complexity. Validation-fitted Isotonic regression restores probability calibration, reducing test Brier score from $0.1654$ to $0.0703$ (beating the $0.0853$ constant-prevalence baseline). Finally, demographic subgroup analysis reveals that precision scales directly with demographic base rates (from $17.7\%$ in young adults to $25.7\%$ in older adults), highlighting that single-threshold screening systems must be interpreted with explicit awareness of population prevalence dynamics.

---

## 1. Introduction & Clinical Screening Context
- Cardiovascular disease (CVD) surveillance and population-level screening using large survey instruments (CDC BRFSS 2015).
- Analysis of Khan et al. (*IEEE Access*, 2024) and their reported benchmarks.
- Key research questions:
  - **RQ1 (Reproducibility):** Are published performance claims (>91% accuracy, >95% precision) reproducible under natural class prevalence without test-set oversampling?
  - **RQ2 (Inductive Bias):** Does structuring tabular inputs into physiological domain groups resolve the inductive bias mismatch of 1D CNNs and GRUs?
  - **RQ3 (Ablation & Gating):** Does dynamic softmax gating over domain representations improve predictive ranking over uniform pooling?
  - **RQ4 (Calibration & Operations):** How do post-hoc probability calibrators and recall-constrained threshold policies behave on untouched test partitions and demographic subgroups?

---

## 2. Dataset Contract & Data Integrity Audit
- **Dataset Contract:** CDC BRFSS 2015 22-column Kaggle-derived subset (253,680 rows, 21 predictors, target `HeartDiseaseorAttack`).
- **Cryptographic Fingerprint:** SHA-256 `953fb23bb131736d7879539a071d3f6d1eaae8d4867ab05335d0ba89b94d5f99`.
- **Target Distribution:** 229,787 negatives (90.58%) vs. 23,893 positives (9.42%).
- **Trivial Baseline:** A naive majority-class classifier predicting 0 for all instances achieves **90.58% accuracy**, but 0% recall and 0% precision.
- **Partitioning:** Stratified 70/15/15 train/valid/test splits preserving natural 9.42% prevalence across all test partitions ($N=38,052$, $P=3,584$).

---

## 3. Related Work & Novelty Assessment
- Literature comparison covering 6 pillars: BRFSS benchmarks, Khan et al. (2024), tabular deep learning (Grinsztajn et al. 2022, Shwartz-Ziv & Armon 2022), feature gating (TabNet, TFT), clinical calibration hierarchy (Van Calster et al. 2016), and algorithmic subgroup disparities (Obermeyer et al. 2019).
- Explicit contribution statement: formal mathematical inconsistency proof, inductive bias demonstration, domain-grouping ablation, and leakage-safe protocol.
- Non-claim: We explicitly refrain from claiming that feature grouping or softmax gating is novel in isolation.

---

## 4. Methodological Audit of Published Work
- **The Prevalence Consistency Finding:** Proof that 96.0% precision and 87.8% recall under 9.42% natural prevalence mathematically requires **98.51% accuracy**, which contradicts the reported 91.7% accuracy. We document that published numbers are mathematically characteristic of an artificially balanced evaluation distribution (~50% prevalence).
- **Inductive Bias Mismatch:** Why passing survey columns through 1D spatial kernels and GRU recurrent states imposes non-existent temporal/spatial physical ordering.
- **ADASYN on Discrete Survey Features:** Empirical demonstration that synthetic interpolation between discrete categorical levels (`Sex=0.42`) corrupts the data manifold, dropping test PR-AUC from 0.3654 to 0.3268.

---

## 5. Domain-Grouped Neural Models & Baseline Ensembles
- **Physiological Grouping:** Partitioning 21 predictors into 4 domain representations:
  1. *Clinical History* (6 features: HighBP, HighChol, CholCheck, Stroke, Diabetes, DiffWalk).
  2. *Lifestyle Factors* (6 features: BMI, Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump).
  3. *Self-Reported Health* (3 features: GenHlth, MentHlth, PhysHlth).
  4. *Healthcare Access & Demographics* (6 features: AnyHealthcare, NoDocbcCost, Sex, Age, Education, Income).
- **Architectures Compared:**
  - `FeatureGroupGatedNet (FGGN)`: Dedicated group MLPs with learned softmax gating network.
  - `UniformGroupNet`: Identical group MLPs with equal $1/4$ uniform pooling (ablation).
  - `EnsCVDD-Net`: Paper's 1D CNN + GRU concatenated hybrid.
  - `HistGradientBoosting`: Tabular histogram tree ensemble (tabular benchmark).
  - `Logistic Regression`: Class-weighted linear baseline.
- **Leakage-Safe Threshold Policy:** Operating threshold $\tau^*$ selected on validation data to meet $\text{Recall} \ge 0.80$ while maximizing precision.

---

## 6. Experimental Protocol & Multi-Seed Benchmark
- 5 fixed random seeds: Seed `42` (exploratory baseline) and Seeds `101`, `2024`, `777`, `9999` (confirmatory).
- All scalers, PBCC feature selection, and class weights fit strictly on training partitions.
- 951,300 persisted test predictions verified with 0.0000000000 discrepancy against summary tables.
- **Direct Multi-Seed Performance Comparison (Mean $\pm$ Std across 5 Seeds on Untouched Test):**
  | Model | PR-AUC (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) | Uncal. Brier Score | Test Recall ($\tau^*$) | Test Precision | Test Specificity |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **HistGradientBoosting** | **0.3717 $\pm$ 0.0067** | **0.8483 $\pm$ 0.0026** | 0.1654 $\pm$ 0.0016 | 79.66% $\pm$ 0.81% | **24.66% $\pm$ 0.42%** | **74.69% $\pm$ 0.68%** |
  | **UniformGroupNet** | 0.3693 $\pm$ 0.0070 | 0.8478 $\pm$ 0.0026 | 0.1672 $\pm$ 0.0035 | 79.75% $\pm$ 0.99% | 24.59% $\pm$ 0.46% | 74.55% $\pm$ 0.84% |
  | **FGGN (Gated Neural)** | 0.3689 $\pm$ 0.0068 | 0.8478 $\pm$ 0.0026 | 0.1659 $\pm$ 0.0046 | 79.78% $\pm$ 0.61% | 24.64% $\pm$ 0.46% | 74.61% $\pm$ 0.75% |
  | **Logistic Regression** | 0.3673 $\pm$ 0.0063 | 0.8462 $\pm$ 0.0029 | 0.1654 $\pm$ 0.0015 | 79.60% $\pm$ 0.95% | 24.41% $\pm$ 0.47% | 74.36% $\pm$ 0.81% |
  | **EnsCVDD-Net** | 0.3563 $\pm$ 0.0055 | 0.8442 $\pm$ 0.0030 | 0.1671 $\pm$ 0.0101 | 79.82% $\pm$ 1.13% | 24.08% $\pm$ 0.44% | 73.82% $\pm$ 0.85% |

---

## 7. Statistical Evaluation & Split Overlap Corrections
- **Repeated Holdout Problem:** Repeated splits share 70% overlapping training data, causing naive $t$-tests to underestimate variance (Dietterich 1998, Nadeau & Bengio 2003).
- **Nadeau & Bengio (2003) Corrected Resampled Inference:**
  $$\hat{\sigma}^2_{\text{corr}} = \left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \hat{S}^2 = (0.200 + 0.214) \hat{S}^2 = 0.414 \hat{S}^2$$
- **Paired PR-AUC Differences Table:**
  | Comparison Pair | Mean $\Delta$ | Standard 95% CI | Standard $p$ | Nadeau-Bengio 95% CI | Nadeau-Bengio $p$ | Interpretation |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **FGGN vs. UniformGroupNet** | -0.000411 | [-0.00135, +0.00053] | $p = 0.293$ | **[-0.00177, +0.00095]** | **$p = 0.448$** | Gating adds no ranking value over uniform pooling (neutral finding). |
  | **FGGN vs. EnsCVDD-Net** | +0.012623 | [+0.00972, +0.01553] | $p = 0.00027$ | **[+0.00844, +0.01680]** | **$p = 0.0011$** | Domain grouping reliably outperforms 1D column convolutions. |
  | **UniformGroupNet vs. EnsCVDD** | +0.013033 | [+0.01044, +0.01563] | $p = 0.00015$ | **[+0.00930, +0.01677]** | **$p = 0.00063$** | Grouping alone drives the performance gain. |
  | **FGGN vs. HistGradientBoosting** | -0.002837 | [-0.00756, +0.00189] | $p = 0.171$ | **[-0.00964, +0.00396]** | **$p = 0.311$** | Gradient boosted trees remain numerically superior. |

---

## 8. Probability Calibration (Van Calster Hierarchy)
- **Constant-Prevalence Baseline Brier Score:** $\bar{y}(1-\bar{y}) = \mathbf{0.085316}$.
- **Uncalibrated Distortion:** Class-weighted loss shifts logits by $\approx \ln(w_+) \approx 2.26$, elevating uncalibrated Brier scores to $\approx 0.166$ and calibration intercepts to $\alpha \approx -2.24$.
- **Validation-Only Post-Hoc Calibration Results (Mean across 5 Seeds):**
  | Model | Baseline Brier | Uncal. Brier | Platt Brier | Isotonic Brier | Uncal. Intercept | Platt Intercept | Platt Slope | Isotonic Intercept | Isotonic Slope |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **HistGradientBoosting** | 0.085316 | 0.165413 | 0.070552 | **0.070274** | -2.236 | -0.011 | 0.992 | -0.058 | 0.963 |
  | **FGGN (Gated)** | 0.085316 | 0.165903 | 0.070662 | **0.070406** | -2.235 | -0.010 | 0.993 | -0.055 | 0.964 |
  | **UniformGroupNet** | 0.085316 | 0.167160 | 0.070637 | **0.070420** | -2.248 | -0.012 | 0.992 | -0.061 | 0.961 |
  | **Logistic Regression** | 0.085316 | 0.165354 | 0.070703 | 0.070581 | -2.246 | -0.006 | 0.996 | -0.058 | 0.962 |
  | **EnsCVDD-Net** | 0.085316 | 0.167127 | 0.071232 | 0.071051 | -2.244 | -0.007 | 0.995 | -0.063 | 0.959 |
- Both Platt scaling (parametric) and Isotonic regression (non-parametric) achieve weak and moderate calibration (slope $\approx 1.0$, intercept $\approx 0.0$), beating the constant-prevalence baseline by $18\%$.

---

## 9. Operational Screening Evaluation & Confusion Counts
- Operating trade-off at validation-constrained $\ge 80\%$ recall:
  - Positives identified: $\approx 2,855$ true positives out of $3,584$ total ($79.66\%$ test recall).
  - False positives: $\approx 8,724$ out of $34,468$ negatives ($74.69\%$ specificity).
  - Screening precision: $\approx 24.66\%$ (roughly 3 false alarms per true case).
- Explaining the trade-off: in low-prevalence screening ($9.4\%$), maintaining high sensitivity inherently produces modest precision due to the large negative volume.

---

## 10. Demographic Subgroup Robustness & Prevalence Sensitivity
- Evaluated across Age brackets, Sex, and Income tiers using predefined BRFSS encodings.
- **Key Empirical Disparity Findings (HistGradientBoosting / FGGN at Global Threshold $\tau^*$):**
  | Demographic Dimension | Subgroup Strata | Subgroup Prevalence | Operating Recall | Operating Precision | Operating Specificity | Subgroup PR-AUC |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Age Strata** | Young Adults (18–44) | **1.38%** | 29.06% | 17.74% | 98.12% | 0.1553 |
  | | Middle-Aged (45–64) | **7.08%** | 68.68% | 22.70% | 82.16% | 0.3165 |
  | | Older Adults (65+) | **17.25%** | 87.72% | 25.71% | 47.16% | 0.4059 |
  | **Sex Strata** | Female | **7.24%** | 72.80% | 22.75% | 80.70% | 0.3228 |
  | | Male | **12.18%** | 84.83% | 26.08% | 66.64% | 0.4058 |
  | **Income Tiers** | Higher Income ($\ge$ $50k) | **5.97%** | 69.84% | 22.68% | 84.88% | 0.3091 |
  | | Middle Income ($25k–<$50k) | **10.74%** | 81.54% | 24.14% | 69.16% | 0.3797 |
  | | Lower Income (<$25k) | **15.96%** | 86.76% | 26.67% | 54.71% | 0.4154 |
- **Prevalence-Driven Precision Dynamics:** Differences in precision and sensitivity across age and sex are mathematically driven by underlying prevalence shifts across demographic strata.
- **Fairness & Safety Boundary:** These findings are documented as demographic subgroup screening dynamics, explicitly noting that they do not constitute proof of algorithmic fairness.

---

## 11. Model Explainability & Attribution
- Learned gate mixing weights: Clinical History (31.8%), Self-Reported Health (30.3%), Lifestyle (25.8%), Demographics/Access (12.1%).
- Extra Trees SHAP: `GenHlth`, `Age`, `HighBP`, `Stroke`, `DiffWalk`, and `BMI`.
- Strict non-causal disclaimer: attributions reflect correlational predictive patterns in cross-sectional survey data, not clinical causal mechanisms.

---

## 12. Threats to Validity, Limitations & Conclusion
- **Self-Reported Survey Data:** BRFSS responses reflect telephone recall, not objective laboratory diagnostics.
- **Cross-Sectional Classification:** Predicts historical condition presence, not prospective future risk.
- **Split Overlap Limitation:** Repeated holdout variance bounded by Nadeau-Bengio correction.
- **Conclusion:** Gradient-boosted trees remain superior on tabular data. Domain grouping resolves 1D sequential convolution flaws, while learned gating adds complexity without ranking gain.
