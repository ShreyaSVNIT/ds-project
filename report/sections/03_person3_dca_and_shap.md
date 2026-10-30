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

