# Theory notes for the report and viva

## 1. Problem setup
Each survey respondent is represented by a feature vector \(x_i \in \mathbb{R}^{21}\), and the target \(y_i \in \{0,1\}\) denotes whether the respondent reported coronary heart disease or myocardial infarction. The task is binary classification: estimate \(p_i = P(y_i=1\mid x_i)\). This label is survey-reported history, not a clinical diagnosis or a prediction of a future heart attack.

## 2. Imbalanced classes and ADASYN
If the minority class occupies a small fraction of the data, a classifier can obtain high accuracy by predicting the majority class and still miss positive cases. ADASYN (Adaptive Synthetic Sampling) assigns more synthetic samples to minority observations whose local neighborhoods are harder to classify. In the paper's broad flow it is used to rebalance the dataset before training.

For a valid holdout experiment, resampling must happen **only after splitting**, on the training partition. Otherwise synthetic points can be influenced by observations that later appear in validation/test. ADASYN also interpolates numeric values; in a feature space containing binary and ordinal survey answers, some synthetic rows can be semantically implausible. Therefore compare ADASYN with class weighting and no resampling rather than assuming it is always best.

## 3. Point-biserial correlation (PBCC)
PBCC measures the association between a numeric feature and a binary variable. Since the target is binary, it is equivalent to Pearson correlation between the numeric feature and the 0/1 label. A common form is

\[
r_{pb}=\frac{\mu_1-\mu_0}{s_x}\sqrt{pq},
\]

where \(\mu_1\) and \(\mu_0\) are the feature means in the positive and negative classes, \(s_x\) is the feature standard deviation under the matching convention, and \(p,q\) are the class proportions. Rank features by \(|r_{pb}|\). Correlation is univariate: it may miss features that are useful only through interactions. The ranking must be estimated on training data only.

## 4. LeNet1D / CNN branch
A one-dimensional convolution computes local learned filters over adjacent feature columns, for example

\[
h_{j,k}=\phi\!\left(b_k+\sum_{c=1}^{C}\sum_{u=0}^{K-1}W_{k,c,u}x_{j+u,c}\right).
\]

A pooling operator reduces the representation; dense layers map it to a probability. The CNN's inductive bias is sensible when adjacent input positions have meaningful locality (e.g. signal samples). For tabular survey columns, adjacency is largely a design choice, so feature-order sensitivity should be acknowledged and may be tested.

## 5. GRU branch
A GRU maintains a hidden state and gates how much previous information is retained. One common convention is

\[
r_t=\sigma(W_r x_t+U_r h_{t-1}),\quad
z_t=\sigma(W_z x_t+U_z h_{t-1}),
\]
\[
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})),\quad
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t.
\]

For an actual time series, \(t\) represents temporal order. Here it represents feature-column order, not time, which is an important limitation of transferring recurrent models to tabular rows.

## 6. EnsCVDD-style parallel hybrid
The LeNet and GRU receive the same feature vector; their intermediate representations are concatenated, \(z=[z_{CNN};z_{GRU}]\), then passed to a dense classification head:

\[
\hat p=\sigma(f_\theta([z_{CNN}(x);z_{GRU}(x)])).
\]

The hypothesis is that the two inductive biases may learn complementary patterns. It must be compared empirically against a simple tabular model and against either branch alone.

## 7. BlCVDD-style blending / stacking
Base models produce prediction features, such as \(m(x)=[p_{CNN}(x),p_{GRU}(x)]\). A meta-learner maps those to the final probability: \(\hat p=g_\phi(m(x))\). To avoid a meta-learner exploiting base models' in-sample overconfidence, generate **out-of-fold** base predictions for training the meta-learner. For the neural paper-style version, a separate validation set trains the meta-model; the test set remains untouched.

## 8. Proposed decision-aware threshold
A predicted probability is not itself a class label. With threshold \(\tau\), predict positive when \(p_i\ge\tau\). The default \(\tau=0.5\) is not necessarily appropriate for an imbalanced screening task. Our validation-only rule selects the **highest** \(\tau\) satisfying \(Recall(\tau)\ge r_{min}\). That seeks the best precision among thresholds that meet the predeclared recall target. It must not be tuned on the test set.

## 9. Metrics
Given true positives (TP), false positives (FP), true negatives (TN), false negatives (FN):

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN},\quad Precision=\frac{TP}{TP+FP},
\]
\[
Recall=Sensitivity=\frac{TP}{TP+FN},\quad Specificity=\frac{TN}{TN+FP},
\]
\[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}.
\]

ROC-AUC summarizes ranking over false-positive rates; PR-AUC / average precision is especially informative when positives are rare. Brier score, \(\frac1N\sum_i(p_i-y_i)^2\), measures probability prediction error. Accuracy, recall, precision and specificity depend on threshold; ROC-AUC and PR-AUC use the continuous scores.

## 10. SHAP
SHAP decomposes a prediction relative to a background/reference distribution into feature attributions. A global summary can help identify which inputs the fitted model relies on; local plots can explain an individual model output. It is not causal proof, not a guarantee of fairness, and not a clinical explanation validated with practitioners.


## 11. Proposed Feature-Group Gated Network (FGGN)
The dataset fields can be grouped by meaning rather than treated as an arbitrary sequence:

- Clinical history: high blood pressure, high cholesterol, cholesterol check, stroke, diabetes, difficulty walking.
- Lifestyle: BMI, smoking, physical activity, fruit/vegetable intake, heavy alcohol consumption.
- Self-reported health: general health, mental-health days, physical-health days.
- Access/demographics: healthcare coverage, cost barrier, sex, age category, education, income.

For each group \(g\), a small encoder computes \(h_g=f_g(x_g)\). A gating network observes all group embeddings and estimates

\[
\alpha=\operatorname{softmax}(W_g[h_1;h_2;\ldots;h_G]+b_g),\qquad \sum_g\alpha_g=1.
\]

The fused vector is \(h=\sum_g\alpha_g h_g\), and the final classifier estimates \(P(y=1\mid x)=\sigma(f(h))\). Unlike the CNN/GRU's arbitrary adjacency/sequence assumption, the group structure is semantically specified. The group weights can show which *groups* influence a fitted prediction; they are not causal effects and should be analyzed across many respondents/seeds, not interpreted from one case. The architecture is a research hypothesis, not a guaranteed improvement.


## 12. Why test-set prevalence matters
The dataset has \(23{,}893/253{,}680\approx 0.0942\) positives. On this untouched distribution, always predicting class 0 yields roughly \(1-0.0942=90.6\%\) accuracy and zero positive-class recall. This is why accuracy alone is insufficient: PR-AUC, recall, specificity and confusion counts must also be reported. Precision depends on prevalence as well as sensitivity/specificity; a precision value measured on a balanced test set is not directly comparable to precision measured at 9.4% prevalence. The paper's Table 14 metrics appear compatible with a near-balanced evaluation distribution, but that is a reproducibility concern to investigate, not conclusive proof of leakage.
