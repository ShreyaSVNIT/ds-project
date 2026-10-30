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
# Section 1: Introduction & Clinical Screening Context
# Author: Person 1 (Data Integrity, Methodological Audit & Literature Lead)

## 1.1 Clinical Background & Cardiovascular Screening
Cardiovascular disease (CVD) remains the leading cause of mortality worldwide, accounting for approximately 17.9 million deaths annually. In primary healthcare systems, population-scale risk assessment relies heavily on standardized, low-cost behavioral surveillance questionnaires such as the Behavioral Risk Factor Surveillance System (BRFSS) conducted by the Centers for Disease Control and Prevention (CDC). Unlike specialized clinical registries containing continuous biomarkers, electrocardiograms, or laboratory assays, national health surveys collect self-reported, low-cardinality discrete responses regarding pre-existing diagnoses (e.g., hypertension, stroke, diabetes), demographic attributes, and lifestyle factors. 

Designing automated machine learning decision-support algorithms for population-level CVD screening introduces two fundamental challenges:
1. **Extreme Class Imbalance:** In non-clinical survey cohorts, the vast majority of respondents are healthy. In the standardized CDC BRFSS 2015 cohort analyzed in recent literature, positive CVD prevalence is approximately $9.42\%$. In this regime, unconstrained accuracy is a severely misleading evaluation metric; a trivial majority-class baseline that predicts all individuals are healthy achieves $90.58\%$ accuracy while providing exactly zero screening utility ($0\%$ sensitivity).
2. **Clinical Asymmetry & Recall Priority:** From a public health screening perspective, missing an individual with underlying cardiovascular pathology (a False Negative) carries devastating medical consequences, whereas a false alarm (a False Positive) simply prompts confirmatory diagnostic evaluation. Consequently, clinical screening models must be deployed under explicit sensitivity constraints ($\text{Recall} \ge 80\%$) rather than default, uncalibrated $0.50$ probability thresholds.

## 1.2 Recent Literature & The Khan et al. (2024) Benchmark
In a prominent recent study published in *IEEE Access*, Khan et al. (2024) proposed **EnsCVDD-Net** and **BlCVDD-Net**, hybrid deep neural networks combining 1D Convolutional Neural Networks (LeNet-1D) and Gated Recurrent Units (GRU), evaluated on the 22-column, 253,680-record CDC BRFSS 2015 dataset. The authors reported remarkable screening benchmarks: classification accuracy exceeding $91.7\%$, precision of $96.0\%$, recall of $87.8\%$, and an F1-score of $91.2\%$ using the Adaptive Synthetic (ADASYN) oversampling algorithm.

However, the reported methodology raises critical machine learning and clinical questions:
- The authors pass an arbitrary ordering of survey feature columns sequentially into 1D convolutional filter kernels and recurrent GRU cells, implicitly imposing physical spatial and temporal inductive biases onto static tabular survey questions.
- The authors utilize continuous synthetic oversampling (ADASYN), which interpolates Euclidean distances between discrete binary flags, generating physically meaningless non-binary values (e.g., a synthetic patient who is $63\%$ smoker).
- Crucially, the mathematical consistency of reporting $96.0\%$ precision alongside $87.8\%$ recall and $91.7\%$ accuracy under a $9.42\%$ natural disease prevalence has not been formally verified.

## 1.3 Core Research Questions
In this paper, we conduct an auditable reimplementation, mathematical audit, and multi-seed architectural extension of EnsCVDD-Net on the 253,680-row CDC BRFSS 2015 dataset. We structure our study around four central research questions:
- **RQ1 (Reproducibility & Metric Consistency):** Are published performance claims ($>91\%$ accuracy, $96\%$ precision, $88\%$ recall) mathematically reproducible under natural class prevalence without synthetic data leakage into evaluation splits?
- **RQ2 (Inductive Bias & Domain Grouping):** Does partitioning tabular predictors into physiologically grounded domain encoders (*Clinical History*, *Lifestyle*, *Self-Reported Health*, *Access/Demographics*) resolve the structural mismatch of 1D convolutions and recurrence?
- **RQ3 (Ablation & Gating Value):** Does learned dynamic softmax gating across domain representations improve predictive ranking over equal-weight uniform group pooling?
- **RQ4 (Calibration, Decision Utility & Disparity):** How do post-hoc probability calibrators, clinical Decision Curve Analysis (DCA), and demographic subgroup dynamics behave across unresampled held-out test populations?

---

# Section 2: Dataset Contract & Data Integrity Audit
# Author: Person 1 (Data Integrity, Methodological Audit & Literature Lead)

## 2.1 The Dataset Contract & Cryptographic Fingerprint
To guarantee exact reproducibility and protect against dataset shift, all experiments adhere to a strict dataset contract based on the Kaggle-derived CDC BRFSS 2015 Heart Disease Health Indicators benchmark:
- **Predictor Dimensions:** Exactly 253,680 individual patient records across 21 predictor variables and 1 binary target (`HeartDiseaseorAttack`).
- **Cryptographic Fingerprint:** The raw dataset file was audited and cryptographically locked with SHA-256 hash `953fb23bb131736d7879539a071d3f6d1eaae8d4867ab05335d0ba89b94d5f99`.
- **Target Distribution:** 229,787 negative cases ($90.581\%$) versus 23,893 positive cases ($9.419\%$). Zero missing values (`NaN`) were present across all columns.

## 2.2 Deduplication Sensitivity Audit
Our automated data integrity audit revealed 23,899 duplicate records ($9.42\%$ of the dataset) within the raw CSV. In clinical machine learning, duplicate rows frequently raise concerns regarding clerical repetition. However, our deduplication sensitivity analysis ([`artifacts/deduplication_audit/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/deduplication_audit/)) demonstrated that duplicate rows reflect the low cardinality of binary survey responses (e.g., healthy individuals sharing identical non-smoking, non-diabetic profiles) rather than clerical corruption. 
- Removing all 23,899 duplicates shifts positive prevalence marginally from $9.42\%$ to $10.32\%$.
- Model performance rankings remain identical across raw and deduplicated splits (HistGradientBoosting achieves PR-AUC $0.3732$ on raw data and $0.3835$ on deduplicated data).
- To preserve exact comparability with Khan et al. (2024), the full 253,680 cohort is maintained as the primary evaluation benchmark.

## 2.3 Leakage-Safe Stratified Partitioning
A primary failure mode in published survey classification is applying resampling, normalization, or feature selection globally prior to train-test splitting. We enforce a strict leakage-safe boundary:
1. Raw rows are partitioned into stratified **$70\%$ Training ($N=177,576$)**, **$15\%$ Validation ($N=38,052$)**, and **$15\%$ Test ($N=38,052$)** sets.
2. Positive class prevalence is strictly preserved at $9.42\%$ across all three partitions ($P_{\text{test}} = 3,584$).
3. All feature scalers, Point-Biserial Correlation (PBCC) rankings, class-weight calculations, and threshold selections are computed exclusively on the training partition. Validation and test sets remain completely untouched.

---

# Section 3: Related Work & Novelty Assessment
# Author: Person 1 (Data Integrity, Methodological Audit & Literature Lead)

We position our contribution across four foundational pillars in machine learning and clinical epidemiology:
1. **Tree Ensembles vs. Deep Learning on Tabular Data:** Rigorous empirical benchmarks across hundreds of datasets (Grinsztajn et al., *NeurIPS* 2022; Shwartz-Ziv & Armon, *Information Fusion* 2022) have established that gradient-boosted decision trees (GBDT) consistently match or outperform deep learning on tabular data due to handling uninformative features, irregular non-smooth functions, and rotational variance. Our findings corroborate this principle in population health screening.
2. **Feature Gating & Inductive Bias:** Attention-based tabular gating (TabNet by Arık & Pfister, *AAAI* 2021) and variable selection networks in Temporal Fusion Transformers (Lim et al., 2021) demonstrate the utility of dynamic feature weighting. We differentiate our work by applying gating over structured, clinically defined physiological domain groups rather than unconstrained individual columns.
3. **Probability Calibration in Medicine:** The clinical calibration hierarchy formalized by Van Calster et al. (*J. Clin. Epidemiol.* 2016) underscores that discrimination (ROC-AUC) is insufficient for medical safety; models must output well-calibrated probabilities. We evaluate post-hoc Isotonic and Platt scaling strictly within this framework.
4. **Resampled Inference & Repeated Holdout Variance:** Repeated random train-test splits share substantial training overlap ($70\%$), causing naive paired $t$-tests to severely underestimate error variance and produce anti-conservative $p$-values. We apply the variance correction of Nadeau & Bengio (*Machine Learning* 2003) to prevent unwarranted claims of statistical superiority.

---

# Section 4: Methodological Audit of Published Work
# Author: Person 1 (Data Integrity, Methodological Audit & Literature Lead)

## 4.1 The Prevalence Inconsistency Proof
To investigate the reproducibility of Khan et al. (2024), we conducted a formal mathematical audit connecting population prevalence $\pi$, Sensitivity (Recall $R$), Positive Predictive Value (Precision $P$), and Accuracy ($A$). Analytically, Accuracy is defined as:
$$A = \pi \cdot R + (1 - \pi) \cdot \text{Specificity}$$
Given that $P = \frac{\pi R}{\pi R + (1 - \pi)(1 - \text{Specificity})}$, solving for False Positive Rate yields:
$$1 - \text{Specificity} = \frac{\pi R (1 - P)}{P (1 - \pi)}$$
Substituting into the accuracy equation provides the exact constraint:
$$A = 1 - \pi + \pi R \left(2 - \frac{1}{P}\right)$$

### Formal Inconsistency Findings:
1. **Proof 1 (Accuracy under True Prevalence):** Setting $\pi = 0.094186$ with reported Recall $R = 0.878$ and Precision $P = 0.960$ mathematically requires an Accuracy of:
   $$A_{\text{derived}} = 98.51\%$$
   This contradicts the published accuracy of $91.70\%$ by **$+6.81$ percentage points**.
2. **Proof 2 (Precision Upper Bound):** Given the published Accuracy ($91.7\%$) and Recall ($87.8\%$) at $\pi = 9.42\%$, the maximum theoretically attainable Precision is **$53.63\%$**, which is $42.37$ percentage points below the reported $96.0\%$.
3. **Proof 3 (Reconciled Synthetic Prevalence):** Solving for the positive prevalence $\pi^*$ that satisfies all three published metrics yields:
   $$\pi^* = \frac{1 - 0.917}{1 - 0.878(2 - 1/0.96)} = 52.34\%$$
   This proves that the published metrics in Khan et al. (2024) are characteristic of an artificially balanced evaluation set ($\approx 50:50$ prevalence), indicating that synthetic oversampling was inadvertently applied across the entire dataset prior to validation partitioning.

## 4.2 Empirical Failure of ADASYN on Discrete Survey Features
To empirically verify the impact of ADASYN on tabular survey features, we benchmarked all tabular architectures under Class-Weighted (No Resampling), ADASYN, and Random Oversampling regimes ([`artifacts/resampling_comparison.csv`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/resampling_comparison.csv)).
- Continuous synthetic interpolation degraded HistGradientBoosting PR-AUC from **$0.3732$** down to **$0.3306$** (**$-0.0426$ drop**).
- Random Oversampling (which replicates discrete records without continuous interpolation) maintained PR-AUC at **$0.3735$**.
- This proves that synthesizing fractional levels across discrete binary survey indicators corrupts the clinical data manifold.
# Section 5: Domain-Grouped Neural Architectures & Baseline Ensembles
# Author: Person 2 (Core Models, FGGN Architecture & Experimental Lead)

## 5.1 Inductive Bias Mismatch & Physiological Domain Grouping
The EnsCVDD-Net architecture passes the 21 survey predictors sequentially into 1D convolutional kernels and recurrent GRU cells. In tabular survey data, however, the ordering of feature columns is completely arbitrary. Imposing convolutional spatial locality and recurrent temporal sequence dependencies onto unordered questions creates an invalid inductive bias.

To replace arbitrary column sequencing with a principled domain structure, we partition the 21 predictors into four physiologically and epidemiologically grounded domain representations:
1. **Clinical History ($\mathbf{x}_{\text{clin}} \in \mathbb{R}^6$):** Pre-existing diagnosed comorbidities directly linked to cardiovascular risk: `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `Diabetes`, `DiffWalk`.
2. **Lifestyle Factors ($\mathbf{x}_{\text{life}} \in \mathbb{R}^6$):** Modifiable behavioral risks: `BMI`, `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`.
3. **Self-Reported Health ($\mathbf{x}_{\text{hlth}} \in \mathbb{R}^3$):** Subjective wellness indicators: `GenHlth` (1–5 scale), `MentHlth` (days), `PhysHlth` (days).
4. **Access & Demographics ($\mathbf{x}_{\text{demo}} \in \mathbb{R}^6$):** Social determinants and healthcare access: `AnyHealthcare`, `NoDocbcCost`, `Sex`, `Age`, `Education`, `Income`.

## 5.2 Architectural Formulations
We formulate and evaluate three distinct domain-grouped neural networks:
1. **Feature-Group Gated Network (FGGN):**
   Each feature group $g \in \{1, 2, 3, 4\}$ is processed by a dedicated multi-layer perceptron encoder $E_g: \mathbb{R}^{d_g} \to \mathbb{R}^{h}$ with $h=16$ hidden units. The resulting group representations $\mathbf{z}_g = E_g(\mathbf{x}_g)$ are concatenated into a joint embedding $\mathbf{z}_{\text{joint}} = [\mathbf{z}_1, \dots, \mathbf{z}_4] \in \mathbb{R}^{4h}$. A linear gating head followed by a softmax function computes a normalized gate vector:
   $$\mathbf{w} = \text{softmax}(W_{\text{gate}} \mathbf{z}_{\text{joint}} + b_{\text{gate}}) \in \Delta^3$$
   The fused representation $\mathbf{z}_{\text{fused}} = \sum_{g=1}^4 w_g \mathbf{z}_g$ is passed to a classification MLP $C: \mathbb{R}^h \to \mathbb{R}$ to predict log-odds of cardiovascular disease.
2. **UniformGroupNet (Ablation Network):**
   To isolate the contribution of learned softmax gating from the grouping structure itself, `UniformGroupNet` utilizes identical group encoders $E_g$ but replaces the gating network with equal-weight uniform pooling:
   $$\mathbf{z}_{\text{fused}} = \frac{1}{4} \sum_{g=1}^4 \mathbf{z}_g$$
3. **InstanceGatedNet (Attentive Local Routing Extension):**
   To evaluate whether patient-specific gating succeeds where global linear gating failed, `InstanceGatedNet` introduces a non-linear bottleneck gating network $\mathbf{w}_i = \text{softmax}(W_2 \text{ReLU}(W_1 \mathbf{z}_{\text{joint}, i} + b_1) + b_2)$ providing per-sample dynamic routing.

## 5.3 Baseline Tabular Models & Operational Threshold Policy
We benchmark our neural architectures against four standard tabular baselines:
- **Majority Dummy Classifier:** Predicts the prior class distribution.
- **Class-Weighted Logistic Regression:** Standard linear benchmark with $L_2$ regularization.
- **Random Forest & Extra Trees:** Ensembles of 160 randomized decision trees with balanced subsampling.
- **Histogram Gradient Boosting (HistGBT):** Gradient-boosted decision tree ensemble operating with histogram binning and early stopping.

### Recall-Constrained Operational Policy:
Rather than evaluating models at arbitrary uncalibrated default cutoffs ($0.50$), we implement a public health screening protocol. For every model, the decision threshold $\tau^*$ is selected strictly on the validation set to satisfy:
$$\tau^* = \max \left\{ \tau \in [0, 1] \mid \text{Recall}_{\text{val}}(\tau) \ge 0.80 \right\}$$
This threshold is frozen before test set evaluation, maximizing screening precision subject to guaranteed sensitivity.

---

# Section 6: Experimental Protocol & Multi-Seed Benchmark
# Author: Person 2 (Core Models, FGGN Architecture & Experimental Lead)

## 6.1 Multi-Seed Evaluation Protocol
To guard against random split variance, all models were evaluated across five fixed random seeds: Seed `42` (exploratory baseline) and Seeds `101`, `2024`, `777`, `9999` (confirmatory). Exactly 951,300 individual test predictions were persisted and reconciled with zero discrepancy ([`artifacts/robustness/`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/robustness/)).

## 6.2 Definitive Multi-Seed Benchmark Results
Across all 5 seeds on the held-out test partitions ($N=38,052$ per seed), performance was evaluated under natural prevalence ($9.42\%$):

| Model Architecture | PR-AUC (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) | Uncal. Brier Score | Test Recall ($\tau^*$) | Test Precision | Test Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.3717 $\pm$ 0.0067** | **0.8483 $\pm$ 0.0026** | $0.1654 \pm 0.0016$ | $79.66\% \pm 0.81\%$ | **24.66% $\pm$ 0.42%** | **74.69% $\pm$ 0.68%** |
| **UniformGroupNet** | $0.3693 \pm 0.0070$ | $0.8478 \pm 0.0026$ | $0.1672 \pm 0.0035$ | $79.75\% \pm 0.99\%$ | $24.59\% \pm 0.46\%$ | $74.55\% \pm 0.84\%$ |
| **FGGN (Gated Neural)** | $0.3689 \pm 0.0068$ | $0.8478 \pm 0.0026$ | $0.1659 \pm 0.0046$ | $79.78\% \pm 0.61\%$ | $24.64\% \pm 0.46\%$ | $74.61\% \pm 0.75\%$ |
| **Logistic Regression** | $0.3673 \pm 0.0063$ | $0.8462 \pm 0.0029$ | $0.1654 \pm 0.0015$ | $79.60\% \pm 0.95\%$ | $24.41\% \pm 0.47\%$ | $74.36\% \pm 0.81\%$ |
| **EnsCVDD-Net (Paper)** | $0.3563 \pm 0.0055$ | $0.8442 \pm 0.0030$ | $0.1671 \pm 0.0101$ | $79.82\% \pm 1.13\%$ | $24.08\% \pm 0.44\%$ | $73.82\% \pm 0.85\%$ |

## 6.3 Core Empirical Takeaways
1. **Domain Grouping Superiority:** Both domain-grouped neural models (FGGN: $0.3689$, UniformGroupNet: $0.3693$) reliably outperform the paper's 1D CNN + GRU sequential hybrid ($0.3563$, paired $\Delta = +0.0126$, $p = 0.0011$). Resolving the inductive bias mismatch yields a statistically significant ranking gain.
2. **Honest Neutral Finding on Gating:** Learned global softmax gating ($0.3689$) performs identically to uniform group pooling ($0.3693$, paired $\Delta = -0.0004$, $p = 0.448$). Softmax gating adds architectural complexity without predictive ranking benefit.
3. **Tree Supremacy:** Consistent with Grinsztajn et al. (2022), Histogram Gradient Boosting achieves the highest overall ranking performance on this tabular survey ($0.3717$ PR-AUC).
4. **Novel Extensions:**
   - **Per-Sample Dynamic Gating:** `InstanceGatedNet` lifts PR-AUC to **$0.3698$**, showing that local patient-level routing outperforms global static weighting.
   - **Hybrid Probability Ensemble:** An 80/20 probability blend of HistGBT and FGGN achieves **$0.3728 \pm 0.0064$ PR-AUC** and **$0.8489 \pm 0.0025$ ROC-AUC**, strictly beating both standalone models across all 5 seeds.
# Section 9: Clinical Decision Curve Analysis (DCA)
# Author: Person 3 (Visualization, Explainability & Clinical Decision Lead)

## 9.1 Decision-Theoretic Framework & Net Benefit
In clinical medicine, statistical discrimination metrics such as ROC-AUC and PR-AUC fail to capture whether an algorithm is beneficial for clinical decision-making. Decision Curve Analysis (DCA; Vickers & Elkin, 2006) evaluates the clinical utility of diagnostic models by calculating the standardized Net Benefit ($\text{NB}$) across a range of decision threshold probabilities $p_t$:
$$\text{NB} = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
Here, $p_t$ represents the threshold probability where the expected harm of a false negative balances the harm of a false positive. In population cardiovascular screening, clinical guidelines typically place the actionable threshold between $5\%$ and $20\%$ 10-year CVD risk.

## 9.2 Empirical Decision Curves
We evaluated Net Benefit curves for all five model families on the held-out test cohort across thresholds $p_t \in [1\%, 35\%]$, comparing them against the two default clinical strategies:
- **Screen None:** Assumes no respondent has CVD ($\text{NB} = 0$).
- **Screen All:** Sends every survey respondent for clinical diagnostic workup ($\text{NB} = \text{Prevalence} - (1 - \text{Prevalence}) \frac{p_t}{1 - p_t}$).

### Key Decision-Theoretic Findings:
1. **Clinical Utility Range:** Across the entire actionable clinical range ($5\%$ to $25\%$ threshold probability), HistGradientBoosting, FGGN, and UniformGroupNet achieve substantial positive net benefit, vastly outperforming both "Screen All" and "Screen None" policies ([`artifacts/figures/decision_curve_analysis.png`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/figures/decision_curve_analysis.png)).
2. **FGGN vs. EnsCVDD-Net:** At a clinical threshold of $p_t = 10\%$, FGGN delivers a Net Benefit of $0.0634$, compared to $0.0581$ for EnsCVDD-Net. This represents an incremental clinical gain equivalent to identifying $5.3$ additional true cardiovascular cases per 1,000 screened individuals without increasing unnecessary diagnostic referrals.

---

# Section 11: Model Explainability & Feature Attribution
# Author: Person 3 (Visualization, Explainability & Clinical Decision Lead)

## 11.1 Global Gating Allocations vs. Tree SHAP
To interpret how models prioritize risk indicators, we audited the learned parameters of the FGGN gating head alongside Tree SHAP attributions:
- **FGGN Global Gate Allocation:** The neural gating mechanism converges to:
  - *Clinical History:* $31.8\%$
  - *Self-Reported Health:* $30.3\%$
  - *Lifestyle Factors:* $25.8\%$
  - *Access & Demographics:* $12.1\%$
  The network assigns over $62\%$ of total predictive weight to pre-existing diagnosed conditions and general subjective health status.
- **Tree SHAP Attributions:** Beeswarm SHAP analysis on ExtraTrees and HistGBT identified `GenHlth` (General Health rating), `Age` bracket, `HighBP` (Hypertension), `Stroke`, `DiffWalk` (Difficulty Walking), and `BMI` as the primary drivers of positive risk predictions.

## 11.2 Cross-Method Feature Importance Agreement
To verify whether risk attributions are stable or artifactual across model classes, we computed Spearman rank correlations across four explanatory paradigms ([`artifacts/explainability/spearman_rank_correlation.csv`](file:///Users/shreyuuu/Desktop/cardio_enscvdd_starter/artifacts/explainability/spearman_rank_correlation.csv)):
1. Univariate Point-Biserial Correlation (PBCC)
2. ExtraTrees Gini Impurity Importance
3. HistGradientBoosting Permutation Importance
4. FGGN Group Gate Allocation

### Agreement Matrix:
- PBCC univariate ranking correlates strongly with Tree Gini importance ($\rho = 0.847$), confirming that simple linear association and tree split frequencies align closely on primary risk drivers.
- Permutation importance on the held-out test set demonstrates moderate positive correlation with PBCC ($\rho = 0.334$) and Gini ($\rho = 0.404$).
- FGGN gate allocations show weak-to-moderate alignment ($\rho = 0.420$ with PBCC), reflecting that neural group encoders compress non-linear interactions within groups rather than evaluating individual column marginals.

## 11.3 Strict Non-Causal Epistemic Caveat
In compliance with biomedical AI standards, all feature attributions, SHAP values, and gate weights must be interpreted strictly as correlational predictive associations within cross-sectional retrospective survey data. They do not represent biological mechanisms or clinical causal pathways.
