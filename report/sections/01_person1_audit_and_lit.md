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

