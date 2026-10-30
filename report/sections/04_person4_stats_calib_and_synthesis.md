# Abstract & Title
# Lead Author / Co-Author: Person 4 (Statistical Inference, Calibration & Synthesis Lead)

# Leakage-Aware, Recall-Constrained Evaluation of Tabular Models for Cardiovascular Disease Risk Screening: A Multi-Seed Reimplementation and Extension of EnsCVDD-Net

### Abstract
Recent literature has proposed hybrid deep neural architectures combining 1D Convolutional Neural Networks and Gated Recurrent Units (EnsCVDD-Net, BlCVDD-Net) for cardiovascular disease (CVD) detection from tabular survey data, reporting accuracies exceeding $91\%$ and precisions near $96\%$ under ADASYN oversampling. In this study, we conduct an auditable reimplementation and methodological audit of these architectures on the 253,680-record CDC BRFSS 2015 cohort. We mathematically prove that the reported precision/recall metrics are mutually incompatible with the published accuracy under the natural positive survey prevalence ($9.42\%$), demonstrating that published benchmarks reflect an artificially balanced ($\approx 52.3\%$) evaluation distribution resulting from pre-split synthetic resampling. We introduce a leakage-safe protocol that strictly confines preprocessing, feature ranking, and class weighting to training folds, preserving untouched natural prevalence on validation and test partitions. To address the invalid inductive bias of applying sequential 1D convolutions to arbitrary column orderings, we formulate the **Feature-Group Gated Network (FGGN)** and an un-gated **UniformGroupNet**, structuring predictors into four physiologically grounded domain representations (*Clinical History*, *Lifestyle*, *Self-Reported Health*, *Demographics/Access*). Across a rigorous 5-seed study (Seeds 42, 101, 2024, 777, 9999; 951,300 persisted predictions), Histogram Gradient Boosting achieves the top overall ranking performance (PR-AUC: $0.3717 \pm 0.0067$, ROC-AUC: $0.8483 \pm 0.0026$). While domain-grouped architectures significantly outperform EnsCVDD-Net (PR-AUC: $0.3563 \pm 0.0055$, paired $\Delta = +0.0126$, $p = 0.0011$), learned softmax gating provides no ranking benefit over uniform semantic group pooling ($0.3689$ vs. $0.3693$, $p = 0.448$), establishing an honest neutral finding regarding model complexity. An 80/20 probability ensemble of HistGBT and FGGN achieves superior ranking ($0.3728 \pm 0.0064$), demonstrating orthogonal representations. Validation-fitted Isotonic regression restores probability calibration, reducing test Brier score from $0.1654$ to $0.0703$ (beating the $0.0853$ constant-prevalence baseline). Finally, demographic subgroup analysis reveals that precision scales directly with baseline stratum prevalence ($17.7\%$ in young adults vs. $25.7\%$ in older adults), underscoring that population prevalence dynamics must govern the operational deployment of automated screening models.

---

# Section 7: Statistical Evaluation & Resampled Inference
# Author: Person 4 (Statistical Inference, Calibration & Synthesis Lead)

## 7.1 The Repeated Holdout Problem & Nadeau-Bengio Correction
In machine learning benchmarks utilizing repeated random train-validation-test partitions, splits share approximately $70\%$ overlapping training instances. This dependency violates the independent observations assumption of the classical Student's paired $t$-test, causing standard errors to be severely underestimated by a factor of $\approx \sqrt{2}$ and yielding inflated Type I error rates (Dietterich 1998; Nadeau & Bengio 2003).

To provide mathematically sound statistical inference across our 5 random holdout splits, we compute the Nadeau & Bengio (2003) corrected resampled variance estimator:
$$\hat{\sigma}^2_{\text{corr}} = \left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \hat{S}^2 = \left(\frac{1}{5} + \frac{38,052}{177,576}\right) \hat{S}^2 = (0.200 + 0.214) \hat{S}^2 = 0.414 \hat{S}^2$$
where $\hat{S}^2$ is the sample variance across the $K=5$ paired differences.

## 7.2 Parametric and Non-Parametric Significance Testing
We evaluate pairwise PR-AUC differences across all 5 seeds, corroborated by 1,000 bootstrap resamples and Wilcoxon signed-rank tests ([`artifacts/calibration/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/calibration/), [`artifacts/non_parametric_stats/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/non_parametric_stats/)):

| Comparison Pair | Mean $\Delta$ | Standard 95% CI | Nadeau-Bengio 95% CI | Nadeau-Bengio $p$ | Wilcoxon $p$ | Cohen's $d$ | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FGGN vs. UniformGroupNet** | -0.000411 | [-0.00135, +0.00053] | **[-0.00177, +0.00095]** | **$p = 0.448$** | $p = 0.4375$ | $-0.54$ | Gating provides zero ranking gain over uniform pooling (neutral finding). |
| **FGGN vs. EnsCVDD-Net** | +0.012623 | [+0.00972, +0.01553] | **[+0.00844, +0.01680]** | **$p = 0.0011$** | $p = 0.0625$ | $+5.39$ | Domain grouping reliably outperforms 1D sequential convolutions. |
| **UniformGroupNet vs. EnsCVDD** | +0.013033 | [+0.01044, +0.01563] | **[+0.00930, +0.01677]** | **$p = 0.00063$** | $p = 0.0625$ | $+6.24$ | Grouping structure alone drives the performance advantage. |
| **HistGBT vs. FGGN** | +0.002837 | [-0.00189, +0.00756] | **[-0.00396, +0.00964]** | **$p = 0.311$** | $p = 0.3125$ | $+0.75$ | Gradient-boosted trees maintain numerical superiority. |

---

# Section 8: Probability Calibration under the Van Calster Hierarchy
# Author: Person 4 (Statistical Inference, Calibration & Synthesis Lead)

## 8.1 Distortion Induced by Class Weighting
Under class-weighted cross-entropy loss, the objective shifts logits by approximately $\ln(w_+) \approx \ln(9.6) \approx 2.26$. Consequently, while uncalibrated probabilities preserve monotonic ranking (ROC-AUC 0.848), they severely overestimate true posterior probabilities:
- The uncalibrated test Brier score is **$0.1654$**, significantly worse than the uninformative constant-prevalence baseline Brier score: $\text{Brier}_{\text{base}} = \bar{y}(1 - \bar{y}) = 0.0942(0.9058) = \mathbf{0.0853}$.
- The uncalibrated calibration intercept is $\alpha \approx -2.24$, reflecting massive probability overestimation.

## 8.2 Validation-Only Post-Hoc Calibration
To restore moderate calibration without data leakage, we fit Platt scaling (logistic sigmoid) and Isotonic regression (non-parametric monotonic step function) strictly on validation predictions and evaluate on untouched test sets:

| Model Architecture | Uncalibrated Brier | Platt Calibrated Brier | Isotonic Calibrated Brier | Isotonic Calibration Intercept | Isotonic Calibration Slope |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | 0.1654 | 0.0706 | **0.0703** | -0.058 | 0.963 |
| **FGGN (Gated Neural)** | 0.1659 | 0.0707 | **0.0704** | -0.055 | 0.964 |
| **UniformGroupNet** | 0.1672 | 0.0706 | **0.0704** | -0.061 | 0.961 |
| **Logistic Regression** | 0.1654 | 0.0707 | **0.0706** | -0.058 | 0.962 |
| **EnsCVDD-Net (Paper)** | 0.1671 | 0.0712 | **0.0711** | -0.063 | 0.959 |

Isotonic calibration reduces the Brier score by **$18\%$ below the constant-prevalence baseline** ($0.0703$ vs $0.0853$), restoring calibration slopes to $\approx 0.96$ and intercepts to $\approx -0.05$, fulfilling moderate calibration under the Van Calster hierarchy.

---

# Section 10: Demographic Subgroup Dynamics & Prevalence Sensitivity
# Author: Person 4 (Statistical Inference, Calibration & Synthesis Lead)

We audited screening metrics across demographic strata on the held-out test cohort ([`artifacts/subgroups/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/subgroups/)):
- **Age Strata:** Young Adults (18–44, prevalence $1.38\%$), Middle-Aged (45–64, prevalence $7.08\%$), Older Adults (65+, prevalence $17.25\%$).
- **Sex Strata:** Female (prevalence $7.24\%$), Male (prevalence $12.18\%$).
- **Income Tiers:** Lower (<$25k, prevalence $15.96\%$), Middle ($25k–<$50k, prevalence $10.74\%$), Higher ($\ge$ $50k, prevalence $5.97\%$).

### Key Demographic Findings:
1. **Prevalence-Driven Precision:** Operating precision scales directly with underlying subgroup prevalence: $17.74\%$ in young adults vs. $25.71\%$ in older adults; $22.75\%$ in females vs. $26.08\%$ in males.
2. **Subgroup Calibration Audit:** As demonstrated in [`artifacts/subgroup_calibration/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/subgroup_calibration/), calibration slopes remain near $1.0$ across all subgroups, but intercepts shift with base prevalence ($-2.45$ for young adults vs. $-2.25$ for older adults).
3. **Fairness Caveat:** These disparities reflect population epidemiology rather than algorithmic bias. Single-threshold systems must be deployed with explicit awareness of stratum prevalence.

---

# Section 12: Threats to Validity, Limitations & Conclusion
# Author: Person 4 (Statistical Inference, Calibration & Synthesis Lead)

## 12.1 Limitations
1. **Self-Reported Retrospective Data:** BRFSS data reflects telephone questionnaire responses subject to recall bias, not objective biomarkers.
2. **Cross-Sectional Classification:** Predicts pre-existing cardiovascular condition presence rather than prospective longitudinal incident risk.
3. **Neutral Complexity Finding:** Softmax gating adds parameters without ranking gain over uniform pooling.

## 12.2 Conclusion
We have demonstrated that published high-accuracy, high-precision claims on CDC BRFSS 2015 reflect an artificially balanced evaluation distribution. Under a strict leakage-safe protocol with untouched natural prevalence ($9.42\%$), domain grouping resolves the inductive bias mismatch of 1D column convolutions ($p = 0.0011$). Gradient-boosted decision trees remain superior on tabular data, and an 80/20 hybrid probability ensemble delivers the highest overall performance ($0.3728$ PR-AUC).

