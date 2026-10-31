# Reproduction + Extension of EnsCVDD-Net / BlCVDD-Net

This starter project is for a course research study based on Khan et al., **“Heart Disease Prediction Using Novel Ensemble and Blending Based Cardiovascular Disease Detection Networks: EnsCVDD-Net and BlCVDD-Net,” IEEE Access (2024)**, DOI: https://doi.org/10.1109/ACCESS.2024.3421241.

It contains (1) a leakage-aware tabular evaluation pipeline, (2) a PyTorch structural reimplementation of the paper's LeNet/GRU/MLP family, (3) metrics and model artifacts, and (4) an optional SHAP script. It is a research/educational prototype, **not a medical device**.

## Working with Antigravity

If using Google Antigravity, open this repository as the workspace. It automatically reads the root `AGENTS.md` as persistent project guidance. Start with `/plan` and the prompt in `ANTIGRAVITY_FIRST_PROMPT.md`; ask it to inspect and plan before executing. The full phased specification is `ANTIGRAVITY_IMPLEMENTATION_PLAN.md`.

## Dataset

The paper uses the **Heart Disease Health Indicators Dataset** derived from CDC's 2015 BRFSS survey: 253,680 rows and 22 columns (the target plus 21 predictors). Target: `HeartDiseaseorAttack`. The positive class is uncommon (about 9–10%), so report class-sensitive metrics rather than accuracy alone.

- Dataset page cited by the paper: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- Official CDC 2015 BRFSS documentation/raw-data landing page: https://www.cdc.gov/brfss/annual_data/annual_2015.html
- A public CSV mirror used by the optional downloader: https://github.com/doguilmak/Heart-Diseaseor-Attack-Classification/blob/main/heart_disease_health_indicators_BRFSS2015.csv

Download the dataset to `data/raw/heart_disease_health_indicators_BRFSS2015.csv` using either:

```bash
python scripts/download_data.py
```

or download the CSV manually from Kaggle and place it at the path above. The dataset is not bundled in this ZIP; check the dataset page's current terms before redistributing it. The official CDC file has a different schema (the full BRFSS survey), so do not pass it to these scripts without first building the paper's 22-column subset.

`SHA256SUMS.txt` contains SHA-256 hashes for tracked repository files only. Generated experiment outputs under `artifacts/` and the local dataset under `data/raw/` are intentionally ignored and are not represented in that manifest. Regenerate the manifest after tracked-file changes with:

```bash
python scripts/generate_checksums.py
```

## Environment

Recommended: Google Colab or a Python 3.10/3.11 environment. For a Mac, a CPU run is possible; the PyTorch models will use MPS if supported. Start with the tabular pipeline before running neural models.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py
```

## Run the experiments

### Paper compilation and sensitivity evaluations

The expanded paper is compiled with Tectonic:

```bash
brew install tectonic
make paper
```

The compiled PDF is written to `report/latex/main.pdf`. The additional evaluations use separate output directories and do not overwrite the primary robustness artifacts:

```bash
make threshold-grid
make duplicate-groups
```

The threshold-grid run selects thresholds on validation predictions for 70%, 80%, 85%, 90%, and 95% recall targets, then evaluates them on untouched test predictions. The duplicate-group run assigns identical predictor vectors to one partition only and records split prevalence in `artifacts/robustness_duplicate_group/split_audit.csv`.

### A. Fast, leakage-aware tabular baseline + proposed extension

```bash
python scripts/run_tabular_experiments.py --data data/raw/heart_disease_health_indicators_BRFSS2015.csv --out artifacts/tabular --resampling none --top-k 21
```

Repeat with the original paper's sampling idea:

```bash
python scripts/run_tabular_experiments.py --data data/raw/heart_disease_health_indicators_BRFSS2015.csv --out artifacts/adasyn --resampling adasyn --top-k 21
```

The held-out validation and test data always keep the original class distribution; resampling happens on training data only. The script saves metrics, fitted models, selected features and run metadata under the chosen output folder. It compares Logistic Regression, Random Forest, Extra Trees, Histogram Gradient Boosting and a diverse stacked blender. It reports (among others) accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC and Brier score.

The **proposed extension** has two parts: (1) a Feature-Group Gated Network (FGGN) that separately encodes clinical history, lifestyle, self-reported health, and access/demographic variables, then learns softmax gate weights for the group embeddings; and (2) leakage-aware, recall-constrained threshold selection. The tabular blender learns out-of-fold base-model predictions for its meta-learner. This changes the operating point; it is not a promise of higher accuracy. The score at threshold 0.50 is still reported as a separate comparison.

To validate the installation without downloading the research dataset or running the full pipeline, use the synthetic smoke check:

```bash
make smoke
```

The generated `synthetic_smoke.csv` and `artifacts/smoke/` outputs are ignored by Git and must not be used as research evidence.

### Why the evaluation itself is a research gap
The source dataset has 23,893 positive cases out of 253,680 (about 9.4%), so a majority-only predictor is already about 90.6% accurate on the original distribution. In Table 14 of the reference paper (PDF page 20), BlCVDD-Net is reported at about 91.7% accuracy, 96.0% precision and 87.8% recall; the base GRU's metrics are all near 90.5%. These metrics look consistent with a near-balanced evaluation set rather than the natural target prevalence. The article does not make the order of ADASYN versus cross-validation splits sufficiently transparent to rule out every concern. This does **not** prove leakage, but it is a strong reason for us to keep the test split untouched and record its prevalence.

Optional: set a smaller `--top-k` such as 15 to test point-biserial feature selection computed on training data only; `--target-recall 0.80` changes the validation target for threshold selection; `--seed` fixes the random seed.

### B. Paper-family neural model structural reimplementation

```bash
python scripts/run_paper_models.py --data data/raw/heart_disease_health_indicators_BRFSS2015.csv --out artifacts/neural --epochs 5 --batch-size 1024 --resampling adasyn
```

This trains standalone LeNet1D and GRU models, a parallel LeNet+GRU hybrid (EnsCVDD-style), a prediction-blending model with an MLP meta-learner (BlCVDD-style), and the proposed Feature-Group Gated Network. The meta-learner is fitted on one half of validation predictions, and its threshold is selected on the other half; the held-out test set remains untouched. Artifacts and metrics are written to `artifacts/neural`.

**Reproduction caveat:** the paper provides the architecture descriptions, tables and reported results but does not provide executable source code in the article. The included neural code is a documented structural reimplementation of those descriptions, not a claim of bit-for-bit reproduction. Dataset preprocessing, architecture details and training budget should be aligned with the exact paper tables if your instructor expects a strict replica.

### C. Explain the tree-based model with SHAP (optional)

After running the tabular script, install SHAP if it is not already installed and run:

```bash
pip install shap
python scripts/explain_shap.py --run-dir artifacts/tabular --model extra_trees_balanced --data data/raw/heart_disease_health_indicators_BRFSS2015.csv
```

It saves a SHAP summary plot on a bounded sample. The neural script also saves average FGGN gate weights in `group_gate_weights.csv`. Explainability plots and gate weights show how the fitted model uses inputs; they do not establish causation or clinical utility.

## Research question and novelty claim

**Primary question:** Does a leakage-aware and recall-constrained ensemble provide a more useful operating point than the paper's accuracy-oriented LeNet/GRU models on this imbalanced, tabular survey dataset?

**Extension components:**
1. Split before fitting feature selection or performing any resampling. Keep the test set untouched and imbalanced.
2. Compare ADASYN with no resampling/class-weighted learning. ADASYN can create fractional values for binary/ordinal questionnaire attributes; document this limitation and do not assume synthetic rows are real people.
3. Compare the paper-family neural models against tabular learners that naturally handle tabular data.
4. Proposed architecture: a feature-group gated network uses separate encoders for clinical history, lifestyle, self-reported health, and access/demographics, then learns a softmax group gate.
5. Use out-of-fold predictions to train the tabular stacking meta-learner; for neural validation blending, separate meta-training and threshold-selection subsets.
6. Select the operating threshold on validation data to maximize precision while meeting a predeclared minimum recall; report the standard 0.50 threshold and recall-constrained threshold separately.
7. Report PR-AUC, recall, specificity, calibration diagnostics (Brier score), confusion matrices, and uncertainty—not just accuracy. Repeat with multiple seeds / stratified folds before making a claim.

The research contribution is the **evaluation protocol and decision-aware extension**, not the assumption that an arbitrary extra model is inherently novel. A +2 percentage-point accuracy gain is a target to test, not a guarantee. If accuracy does not improve, a measured improvement in recall at similar specificity, calibration, runtime, or stability can still form a valid result if pre-specified and honestly reported.

## What to include in the final course submission

- **Research paper/report:** problem setup; paper's method; dataset and class distribution; exact split and leakage controls; model equations/architecture; experiment matrix; results with uncertainty; ablations; SHAP interpretation; limitations and reproducibility.
- **Code:** this repository, `requirements.txt`, fixed seeds, data download/preprocessing scripts, configuration/metadata, saved model weights, metrics and plots. Never commit a large dataset without checking redistribution terms.
- **PPT:** motivation, baseline paper, gap, hypothesis/novelty, data and leakage controls, architecture diagram, experiment matrix, results, ablations, interpretability, limitations and conclusion.

## Important limitations

- This dataset is a self-reported U.S. survey and targets ever reporting coronary heart disease or myocardial infarction, not a clinical diagnosis from current tests.
- Age, diabetes and other encoded survey columns are ordinal or categorical in meaning; a synthetic interpolation can be semantically invalid.
- A higher score on this one dataset does not establish generalization to other populations or future cases.
- The CNN/GRU treat a fixed ordering of columns like a 1-D signal/sequence. There is no natural temporal sequence in these rows; the comparison to tabular methods is part of the research question.
- This is not for patient-level medical decisions.
