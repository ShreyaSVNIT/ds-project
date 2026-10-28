# Paper Reproduction & Discrepancy Audit Notes
## Quantitative Comparison of Published Claims vs. Leakage-Safe Multi-Seed Study

### 1. Published Paper Claims (Khan et al., IEEE Access 2024, Table 14)
Khan et al. report the following metrics for 30-epoch models evaluated using 10-fold cross-validation with ADASYN oversampling on the Kaggle-derived 22-column BRFSS 2015 subset:

| Model | Reported Accuracy | Reported Precision | Reported Recall | Reported F1-Score |
|---|---|---|---|---|
| **LeNet1D** | 89.8% | 94.8% | 84.8% | 88.9% |
| **GRU** | 90.5% | 95.5% | 85.5% | 89.6% |
| **EnsCVDD-Net** | 91.2% | 95.8% | 86.8% | 90.5% |
| **BlCVDD-Net** | 91.7% | 96.0% | 87.8% | 91.2% |

---

### 2. Metric & Prevalence Consistency Analysis
The CDC BRFSS 2015 22-column dataset contains **23,893 positive cases out of 253,680 rows (prevalence $p = 0.094186 \approx 9.42\%$)**.

#### 2.1 The Trivial Baseline
A naive majority-class predictor assigning class 0 to every respondent achieves **90.581% accuracy** with zero recall and zero precision. Therefore, the published 91.7% accuracy represents only a nominal +1.1% increase over guessing negative for every patient.

#### 2.2 Mathematical Consistency Audit of Published Metrics
Under standard definitions of confusion matrix rates, if true positive prevalence is $p = 0.094186$, recall is $r = 0.878$, and precision is $q = 0.960$:
1. Positive fraction correctly identified:
   $$\text{TP} = p \cdot r = 0.094186 \times 0.878 = 0.082695$$
2. False negatives:
   $$\text{FN} = p \cdot (1 - r) = 0.094186 \times 0.122 = 0.011491$$
3. From $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = q$, the required false positive fraction is:
   $$\text{FP} = \text{TP} \cdot \frac{1 - q}{q} = 0.082695 \times \frac{0.04}{0.96} = 0.003446$$
4. Since total negative prevalence is $1 - p = 0.905814$, the true negative fraction is:
   $$\text{TN} = (1 - p) - \text{FP} = 0.905814 - 0.003446 = 0.902368$$
5. Resulting theoretical accuracy on this population:
   $$\text{Accuracy} = \text{TP} + \text{TN} = 0.082695 + 0.902368 = \mathbf{98.51\%}$$

**Scientific Finding:** The reported accuracy of **91.7%** is mathematically incompatible with simultaneously achieving **96.0% precision** and **87.8% recall** on a 9.42% prevalence population. 

Conversely, if an evaluation set were artificially balanced to 50% positive prevalence ($p = 0.5$):
$$\text{Accuracy} = 0.5 \cdot 0.878 + 0.5 \cdot \text{TNR} = 0.917 \implies \text{TNR} \approx 0.956$$
$$\text{Precision} = \frac{0.5 \cdot 0.878}{0.5 \cdot 0.878 + 0.5 \cdot (1 - 0.956)} = \frac{0.439}{0.439 + 0.022} \approx \mathbf{95.2\%}$$

**Objective Framing:** We do not claim proof of intentional manipulation or prove that test folds were oversampled, as the original paper does not release executable code. However, we document this mathematical discrepancy: the reported metrics in Table 14 are mathematically characteristic of an artificially balanced evaluation distribution rather than the natural 9.42% survey distribution.

---

### 3. Rigorous Multi-Seed Holdout Evaluation (5 Seeds, 9.42% Natural Test Prevalence)

Evaluated across 5 fixed seeds (`42` exploratory; `101`, `2024`, `777`, `9999` confirmatory) on untouched test sets ($N=38,052$, $P=3,584$, $N=34,468$ per split). All scalers, feature selection, and class weighting fit strictly on training data. Persisted test predictions (951,300 rows) verified with zero discrepancy.

| Model | Model Description | PR-AUC (Avg. Prec.) Mean $\pm$ Std | ROC-AUC Mean $\pm$ Std | Operating Test Recall (Target $\ge 80\%$) | Operating Test Precision Mean $\pm$ Std | Operating Test Specificity Mean $\pm$ Std | Brier Score Mean $\pm$ Std |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | Tabular Gradient Boosting | **0.3717 $\pm$ 0.0067** | **0.8483 $\pm$ 0.0026** | 79.66% $\pm$ 0.81% | **24.66% $\pm$ 0.42%** | **74.69% $\pm$ 0.68%** | **0.1654 $\pm$ 0.0016** |
| **UniformGroupNet** | Semantic Group Fusion Ablation | 0.3693 $\pm$ 0.0070 | 0.8478 $\pm$ 0.0026 | 79.75% $\pm$ 0.99% | 24.59% $\pm$ 0.46% | 74.55% $\pm$ 0.84% | 0.1672 $\pm$ 0.0035 |
| **FeatureGroupGatedNet (FGGN)** | Proposed Domain-Gated Neural Net | 0.3689 $\pm$ 0.0068 | 0.8478 $\pm$ 0.0026 | 79.78% $\pm$ 0.61% | 24.64% $\pm$ 0.46% | 74.61% $\pm$ 0.75% | 0.1659 $\pm$ 0.0046 |
| **Logistic Regression** | Class-Weighted Linear Baseline | 0.3673 $\pm$ 0.0063 | 0.8462 $\pm$ 0.0029 | 79.60% $\pm$ 0.95% | 24.41% $\pm$ 0.47% | 74.36% $\pm$ 0.81% | 0.1654 $\pm$ 0.0015 |
| **EnsCVDD-Net** | Paper LeNet1D + GRU Hybrid | 0.3563 $\pm$ 0.0055 | 0.8442 $\pm$ 0.0030 | 79.82% $\pm$ 1.13% | 24.08% $\pm$ 0.44% | 73.82% $\pm$ 0.85% | 0.1671 $\pm$ 0.0101 |

---

### 4. Empirical Takeaways & Resolution of Findings

1. **Resolution of FGGN Discrepancy:**
   - In earlier logs, FGGN with native class-weighting achieved PR-AUC 0.3654 / ROC-AUC 0.8467 on seed 42. In the ADASYN experiment, FGGN achieved PR-AUC 0.3212 / ROC-AUC 0.8291.
   - The discrepancy occurred because the earlier report summary cited the class-weighted ablation row while the checkpoint in `artifacts/neural_full/models/` was from the ADASYN run.
   - In the multi-seed protocol with class-weighting across 5 seeds, FGGN achieves a mean PR-AUC of **0.3689 $\pm$ 0.0068**, fully confirming the class-weighted performance.
2. **FGGN is NOT the Winning Model:**
   - `HistGradientBoosting` achieves the highest mean PR-AUC (**0.3717**).
   - `UniformGroupNet` (**0.3693**) and `FGGN` (**0.3689**) perform identically within statistical error ($\sigma \approx 0.007$). Learned gating does not improve ranking over uniform semantic pooling.
3. **Domain Grouping Outperforms 1D Sequence Hybrids:**
   - Both `FGGN` ($0.3689$) and `UniformGroupNet` ($0.3693$) consistently outperform `EnsCVDD-Net` ($0.3563$) across all 5 seeds by approximately $+0.013$ PR-AUC points. Structuring tabular inputs into clinical domains is demonstrably superior to arbitrary 1D column convolutions and recurrent recurrence.
4. **Validation-Only Decision Thresholding is Robust:**
   - Across all 5 seeds, selecting operating thresholds on validation data to meet $\ge 80\%$ recall consistently delivered test recall of **79.6% to 79.8%**, test precision of **24.1% to 24.7%**, and test specificity of **73.8% to 74.7%**.
5. **Probability Calibration (Van Calster Hierarchy):**
   - Constant-prevalence Brier baseline is $\mathbf{0.085316}$.
   - Uncalibrated class-weighted predictions have elevated Brier scores ($\approx 0.165–0.167$) due to logit shift by $\ln(w_+)$.
   - Validation-fitted Platt scaling and Isotonic regression reduce test Brier scores to $\mathbf{0.0706}$ and $\mathbf{0.0703}$ respectively, beating the prevalence baseline.
6. **Paired Statistical Differences & Split Overlap Caveat:**
   - FGGN vs. UniformGroupNet: mean $\Delta = -0.000411$, 95% CI $[-0.001353, +0.000531]$, $p = 0.293$. Confirms learned gating has no ranking benefit over uniform pooling.
   - FGGN vs. EnsCVDD-Net: mean $\Delta = +0.012623$, 95% CI $[+0.009717, +0.015528]$, $p = 0.00027$.
   - Repeated holdouts share 70% training data overlap; Nadeau-Bengio corrected resampled variance accounts for this correlation (Dietterich 1998, Nadeau & Bengio 2003). Standard error and intervals reflect split sensitivity rather than independent draws.
