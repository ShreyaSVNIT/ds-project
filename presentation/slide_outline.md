# Presentation Slide Outline (12 Slides)
## Title: Leakage-Aware Evaluation of Tabular Models for Classifying Self-Reported Heart Disease History in BRFSS 2015
### An Auditable Reimplementation and Multi-Seed Extension of EnsCVDD-Net

---

### Slide 1: Title & Research Overview
- **Title:** Leakage-Aware Evaluation of Tabular Models for Classifying Self-Reported Heart Disease History in BRFSS 2015
- **Subtitle:** An Auditable Reimplementation and Multi-Seed Extension of EnsCVDD-Net
- **Context:** Course Research Prototype — Strictly Not a Clinical Diagnostic Tool
- **Core Message:** Transparent multi-seed benchmark comparing gradient-boosted trees, sequential neural networks, and domain-grouped architectures under strict natural prevalence.

### Slide 2: The Reference Paper & Published Claims
- **Reference Paper:** Khan et al. (*IEEE Access*, 2024, DOI: `10.1109/ACCESS.2024.3421241`)
- **Proposed Architectures:** EnsCVDD-Net (1D LeNet + GRU hybrid) & BlCVDD-Net (MLP meta-blender).
- **Reported Claims:** 91.7% Accuracy, 96.0% Precision, 87.8% Recall, 91.2% F1 on BRFSS 2015.
- **The Research Question:** How do these models perform under true population prevalence and rigorous leakage-isolated multi-seed evaluation?

### Slide 3: Dataset Reality & The Prevalence Consistency Finding
- **Dataset:** CDC BRFSS 2015 22-column subset (253,680 rows, 21 predictors, target `HeartDiseaseorAttack`).
- **The Class Imbalance Reality:** 23,893 positive cases (**9.42% natural prevalence**).
- **The Trivial Baseline:** A majority-class classifier predicting negative for all rows achieves **90.58% accuracy**!
- **The Mathematical Reality Check:**
  - Achieving 96% precision with 87.8% recall on a 9.42% prevalence population mathematically yields **98.51% accuracy**.
  - The published 91.7% accuracy is characteristic of an artificially balanced evaluation fold (~50% prevalence), not the natural survey population.

### Slide 4: Methodological Vulnerabilities in Literature
- **Inductive Bias Mismatch:** Passing tabular survey columns sequentially through 1D CNN kernels and GRU hidden states imposes artificial spatial and temporal physics on unordered features.
- **ADASYN Interpolation Artifacts:** Synthetic interpolation between discrete survey categories creates nonsensical fractional points (`Sex=0.42`, `Smoker=0.73`), degrading test PR-AUC from 0.3654 to 0.3268.
- **OOF Leakage Risk:** Oversampling before cross-validation contaminates held-out evaluation folds.

### Slide 5: Investigational Architecture: Domain-Grouped Neural Networks
- **Physiological Grouping:** Replaces 1D column ordering with 4 semantic domain representations:
  1. *Clinical History* (HighBP, HighChol, CholCheck, Stroke, Diabetes, DiffWalk).
  2. *Lifestyle* (BMI, Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump).
  3. *Self-Reported Health* (GenHlth, MentHlth, PhysHlth).
  4. *Access & Demographics* (Healthcare, Cost, Sex, Age, Education, Income).
- **Architectures Evaluated:**
  - `FeatureGroupGatedNet (FGGN)`: Dedicated group MLPs with dynamic softmax gating network.
  - `UniformGroupNet (Ablation)`: Equal $1/4$ uniform pooling over group embeddings (no learned gates).

### Slide 6: Frozen Multi-Seed Evaluation Protocol
- **5 Fixed Random Seeds:** `42` (exploratory baseline); `101`, `2024`, `777`, `9999` (confirmatory).
- **Stratified 70/15/15 Splits:** Untouched 9.42% natural prevalence preserved in test partitions ($N=38,052$, $P=3,584$).
- **Strict Leakage Isolation:** Preprocessing, PBCC ranking, class-weighting, threshold selection, and calibrators fit strictly on training/validation folds.
- **Persisted Predictions:** 951,300 test predictions persisted and reconciled with zero discrepancy.

### Slide 7: Multi-Seed Benchmark Results (Mean $\pm$ Std across 5 Seeds)
- **Performance at 9.42% Natural Test Prevalence:**
  - *HistGradientBoosting (Tabular SOTA):* **PR-AUC 0.3717 $\pm$ 0.0067**, **ROC-AUC 0.8483 $\pm$ 0.0026**.
  - *UniformGroupNet (Group Ablation):* **PR-AUC 0.3693 $\pm$ 0.0070**, **ROC-AUC 0.8478 $\pm$ 0.0026**.
  - *FGGN (Gated Neural Net):* **PR-AUC 0.3689 $\pm$ 0.0068**, **ROC-AUC 0.8478 $\pm$ 0.0026**.
  - *Logistic Regression (Linear Baseline):* **PR-AUC 0.3673 $\pm$ 0.0063**, **ROC-AUC 0.8462 $\pm$ 0.0029**.
  - *EnsCVDD-Net (Paper Hybrid):* **PR-AUC 0.3563 $\pm$ 0.0055**, **ROC-AUC 0.8442 $\pm$ 0.0030**.
- **Key Takeaways:** Gradient-boosted trees remain superior. Domain grouping outperforms 1D sequence models across all seeds.

### Slide 8: Statistical Comparisons & Novelty Assessment
- **FGGN vs. UniformGroupNet (Learned Gating Value):**
  - Mean $\Delta = -0.000411$, Standard 95% CI: $[-0.00135, +0.00053]$, Nadeau-Bengio 95% CI: $[-0.00177, +0.00095]$, $p = 0.448$.
  - *Finding:* Learned softmax gating adds complexity with **zero ranking benefit** over uniform pooling.
- **FGGN vs. EnsCVDD-Net (Domain Grouping vs. 1D Sequence):**
  - Mean $\Delta = +0.012623$, Nadeau-Bengio 95% CI: $[+0.00844, +0.01680]$, $p = 0.0011$.
  - *Finding:* Semantic domain grouping reliably outperforms arbitrary 1D column convolutions across all splits.
- **Nadeau & Bengio (2003) Split Overlap Correction:**
  - Repeated holdout splits share 70% training data; corrected variance widens intervals to prevent overstating significance.

### Slide 9: Probability Calibration & Risk Spread (Van Calster Hierarchy)
- **Constant-Prevalence Baseline Brier Score:** $\bar{y}(1-\bar{y}) = \mathbf{0.085316}$.
- **Uncalibrated Distortion:** Class-weighting shifts output logits by $\ln(w_+) \approx 2.26$, elevating uncalibrated Brier score to $\approx 0.166$ and intercept to $\alpha \approx -2.24$.
- **Validation-Fitted Post-Hoc Calibration:**
  - Platt scaling restores calibration-in-the-large (intercept $\approx -0.010$, slope $\approx 0.993$), Brier score $\mathbf{0.0706}$.
  - Isotonic regression achieves lowest test Brier score of $\mathbf{0.0703}$ (18% below prevalence baseline).
- **Key Takeaway:** Uncalibrated class-weighted predictions cannot be interpreted as probabilities; validation-fitted calibration restores reliable posterior risks.

### Slide 10: Operational Decision-Constrained Screening Performance
- Under validation recall constraint ($\ge 80.0\%$ sensitivity):
  - *HistGradientBoosting:* **79.66% $\pm$ 0.81% Test Recall**, **24.66% $\pm$ 0.42% Precision**, **74.69% $\pm$ 0.68% Specificity**.
  - *FGGN:* **79.78% $\pm$ 0.61% Test Recall**, **24.64% $\pm$ 0.46% Precision**, **74.61% $\pm$ 0.75% Specificity**.
  - *Confusion Matrix Reality:* $\approx 2,855$ true positives vs. $\approx 8,724$ false positives per 38,052 individuals.
- **Operational Reality:** In low-prevalence screening ($9.4\%$), maintaining ~80% sensitivity requires accepting ~25% precision (roughly 3 false alarms per true case).

### Slide 11: Demographic Subgroup Robustness & Prevalence Sensitivity
- Evaluated across Age strata, Sex, and Income tiers using predefined BRFSS encodings.
- **Empirical Disparity Findings (Global Threshold $\tau^*$):**
  - *Young Adults (18–44, 1.38% Prev):* Recall 29.1%, Precision 17.7%, Specificity 98.1%.
  - *Older Adults (65+, 17.25% Prev):* Recall 87.7%, Precision 25.7%, Specificity 47.2%.
  - *Sex Dynamics:* Males (12.2% Prev, Recall 84.8%, Prec 26.1%) vs. Females (7.2% Prev, Recall 72.8%, Prec 22.8%).
- **Crucial Insights:**
  - Precision and operating recall naturally scale with demographic prevalence under a fixed threshold.
  - Subgroup performance analysis does **not** constitute proof of algorithmic fairness.

### Slide 12: Ethical Disclaimers, Limitations & Conclusions
- **Essential Clinical Disclaimers:**
  - Target `HeartDiseaseorAttack` is **self-reported survey history**, not physician-verified clinical diagnostics.
  - Cross-sectional classification of past events, not prospective future risk forecasting.
  - Gate weights and SHAP attributions represent predictive associations, not biological causes.
- **Final Conclusions:**
  1. Gradient-boosted decision trees remain the superior tabular baseline.
  2. Domain grouping outperforms 1D sequential convolutions/recurrence on tabular survey data.
  3. Learned softmax gating adds complexity without ranking gain (honest negative finding).
  4. Post-hoc calibration is essential when using class-weighted losses in population screening.
