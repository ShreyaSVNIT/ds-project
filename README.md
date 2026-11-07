# Reproduction + Extension of EnsCVDD-Net / BlCVDD-Net

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: Reproducibility](https://img.shields.io/badge/Research-Empirical%20Audit-success.svg)](#key-research-findings--results-summary)

This repository contains the rigorous course research study and empirical evaluation based on:
> Khan et al., **“Heart Disease Prediction Using Novel Ensemble and Blending Based Cardiovascular Disease Detection Networks: EnsCVDD-Net and BlCVDD-Net,” IEEE Access (2024)**, DOI: [10.1109/ACCESS.2024.3421241](https://doi.org/10.1109/ACCESS.2024.3421241).

It provides:
1. **Leakage-aware tabular evaluation pipeline** with pre-split feature selection and untouched, imbalanced test distributions.
2. **PyTorch structural reimplementation** of the paper's LeNet1D, GRU, EnsCVDD-Net, and BlCVDD-Net architectures.
3. **Novel Extension:** Feature-Group Gated Network (FGGN), recall-constrained threshold optimization, and probability calibration.
4. **Empirical Audit & Statistical Testing:** 5-seed confirmatory study ($N=951,300$ reconciled test predictions), Nadeau-Bengio corrected variance, Platt/Isotonic calibration, demographic fairness subgroup audits, and clinical Decision Curve Analysis (DCA).
5. **Publication-Ready Deliverables:** Complete IEEE-style LaTeX paper (`report/latex/main.pdf`) and final presentation slide deck (`presentation/FGGN_Cardiovascular_Research_Presentation_Final.pptx`).

> [!WARNING]
> This repository is an academic/educational research prototype for machine learning methodology evaluation, **not a clinical decision support system or medical device**.

---

## Directory Structure

```text
ds-project/
├── configs/                # Experiment configurations (baseline.yaml)
├── data/                   # Dataset directory (.gitignore excludes large raw CSVs)
│   └── raw/                # Target folder for CDC BRFSS dataset
├── presentation/           # Presentation materials
│   ├── FGGN_Cardiovascular_Research_Presentation_Final.pptx # Final presentation slide deck
│   └── slide_outline.md    # Detailed presentation slide script
├── report/                 # Full research report and paper sources
│   ├── latex/              # IEEE Access LaTeX source (main.tex, references.bib)
│   │   ├── figures/        # 12 publication-quality figures
│   │   └── main.pdf        # Compiled two-column research paper PDF
│   ├── FULL_RESEARCH_PAPER.md    # Comprehensive markdown report
│   └── FULL_RESEARCH_PAPER.html  # Standalone HTML report
├── research/               # Related literature matrices & notes
├── scripts/                # End-to-end execution and reproducibility scripts
│   ├── audit_dataset.py                   # Data audit, schema validation & split hashing
│   ├── download_data.py                   # Automated dataset retriever
│   ├── evaluate_calibration_and_stats.py  # Platt/Isotonic calibration & corrected CIs
│   ├── evaluate_dca.py                    # Clinical Decision Curve Analysis
│   ├── evaluate_subgroups.py              # Demographic slice & fairness analysis
│   ├── generate_checksums.py              # Cryptographic manifest generator
│   ├── run_paper_models.py                # PyTorch paper architectures & FGGN training
│   ├── run_tabular_experiments.py         # Leakage-safe tabular baselines & stacking
│   ├── run_multiseed_robustness.py        # 5-seed confirmatory protocol
│   └── verify_and_reconcile_metrics.py    # Zero-discrepancy prediction reconciler
├── src/                    # Core modular Python library
│   └── cardio/             # Data splits, preprocessing, metrics, tabular & neural models
├── tests/                  # Pytest test suite (13 unit and integration tests)
├── AGENTS.md               # Persistent instructions and architecture contracts
├── IMPLEMENTATION_STATUS.md # Experiment log, milestone tracking, and audit records
├── Makefile                # Reproducibility commands
├── README.md               # Repository documentation
├── requirements.txt        # Python package dependencies
└── SHA256SUMS.txt          # Cryptographic SHA-256 manifest of tracked files
```

---

## Key Research Findings & Results Summary

From our multi-seed confirmatory evaluation (seeds 42, 101, 2024, 777, 9999; $N=951,300$ held-out test predictions):

| Model Architecture | Model Family | PR-AUC (Mean ± SD) | ROC-AUC (Mean ± SD) | Brier Score | Operating Recall @ 80% Target |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** *(Top Performer)* | Tabular GBDT | **0.3717 ± 0.0067** | **0.8492 ± 0.0031** | **0.0694** | **80.1%** (Prec: 24.3%) |
| **UniformGroupNet** | Neural Group Pooling | 0.3693 ± 0.0070 | 0.8471 ± 0.0033 | 0.0712 | 80.0% (Prec: 23.8%) |
| **FGGN (Proposed)** *(Class-weighted)* | Neural Gated Network | 0.3689 ± 0.0068 | 0.8468 ± 0.0032 | 0.0715 | 80.0% (Prec: 23.7%) |
| **Logistic Regression** *(Balanced)* | Linear Baseline | 0.3673 ± 0.0063 | 0.8449 ± 0.0030 | 0.0721 | 80.0% (Prec: 23.2%) |
| **EnsCVDD-Net** *(Paper Hybrid)* | CNN + GRU | 0.3563 ± 0.0055 | 0.8354 ± 0.0035 | 0.0748 | 79.8% (Prec: 22.1%) |
| **Majority Baseline** | Trivial Prior | 0.0942 (Prevalence) | 0.5000 | 0.0853 | 0.0% (Trivial) |

### Critical Methodological Insights:
1. **Prevalence Audit:** The target prevalence is 9.42% (23,893 positive cases out of 253,680). The reference paper's claim of ~91.7% accuracy with 96% precision is mathematically incompatible with the natural distribution, indicating that earlier work evaluated models on artificially resampled test splits. In this work, the test split is kept untouched and imbalanced.
2. **Resampling Pitfall:** ADASYN oversampling degrades PR-AUC on tabular survey data (0.3268 vs. 0.3689 for class-weighted loss) because synthetic interpolation creates semantically invalid fractions on ordinal/binary questionnaire features.
3. **Tabular vs. Deep Learning:** Histogram Gradient Boosting outperforms deep neural architectures on this tabular dataset, while requiring orders of magnitude less training compute.

---

## Dataset

The study uses the **Heart Disease Health Indicators Dataset** derived from CDC's 2015 BRFSS survey (253,680 rows, 22 columns). Target: `HeartDiseaseorAttack`.

- **Kaggle Page:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- **Official CDC 2015 BRFSS Page:** https://www.cdc.gov/brfss/annual_data/annual_2015.html
- **Direct CSV Mirror:** https://github.com/doguilmak/Heart-Diseaseor-Attack-Classification/blob/main/heart_disease_health_indicators_BRFSS2015.csv

Download and place the dataset at `data/raw/heart_disease_health_indicators_BRFSS2015.csv` via:

```bash
python scripts/download_data.py
```

`SHA256SUMS.txt` tracks cryptographic hashes of all source and configuration files. Regenerate after modifications with:

```bash
python scripts/generate_checksums.py
```

---

## Installation & Environment Setup

Python 3.10, 3.11, 3.12, or 3.13 is recommended.

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Reproducibility Commands (`Makefile` & CLI)

A comprehensive `Makefile` automates all phases:

| Command | CLI Equivalent | Description |
| :--- | :--- | :--- |
| `make smoke` | `python scripts/generate_synthetic_smoke.py ...` | Fast smoke test verifying end-to-end tabular execution on synthetic data |
| `make audit` | `python scripts/audit_dataset.py ...` | Cryptographic audit, schema checks, and prevalence verification |
| `make tabular` | `python scripts/run_tabular_experiments.py ...` | Runs all leakage-safe tabular baselines and stacking meta-blender |
| `make neural` | `python scripts/run_paper_models.py ...` | Runs LeNet1D, GRU, EnsCVDD-Net, BlCVDD-Net, and proposed FGGN |
| `make ablations` | `python scripts/run_ablation_study.py ...` | Evaluates gating mechanisms and resampling (ADASYN vs class-weighting) |
| `make robustness` | `python scripts/run_multiseed_robustness.py ...` | 5-seed confirmatory study across seeds 42, 101, 2024, 777, 9999 |
| `make calibration` | `python scripts/evaluate_calibration_and_stats.py ...` | Fits Platt/Isotonic calibrators on validation; evaluates Brier scores |
| `make subgroups` | `python scripts/evaluate_subgroups.py ...` | Evaluates demographic fairness slices across Age, Sex, and Income |
| `make dca` | `python scripts/evaluate_dca.py` | Computes clinical net benefit curves across decision thresholds |
| `make shap` | `python scripts/evaluate_shap_agreement.py` | Computes TreeSHAP feature attributions and agreement ranking |
| `make paper` | `tectonic -X compile report/latex/main.tex` | Compiles two-column research paper into `report/latex/main.pdf` |
| `make test` | `pytest -v tests/` | Executes full unit and integration test suite |
| `make all` | Full pipeline execution | Executes complete reproduction and extension workflow |

### Fast Synthetic Smoke Test
To verify code execution without waiting for the full 250k-row training run:

```bash
make smoke
# or directly:
python scripts/generate_synthetic_smoke.py --out data/raw/synthetic_smoke.csv
python scripts/run_tabular_experiments.py --data data/raw/synthetic_smoke.csv --out artifacts/smoke --resampling none --top-k 21 --quick
```

---

## Development and Agent Guidance

When inspecting or extending this codebase in an agentic environment (such as Google Antigravity or VS Code):
- Read `AGENTS.md` for persistent project guidance, coding standards, and reproducibility boundaries.
- Inspect `IMPLEMENTATION_STATUS.md` for completed milestones, historical experiment results, and verified metric tables.
- Refer to `PROFESSOR_DOUBTS_AND_ROADMAP.md` and `PROJECT_PLAN.md` for background research context and technical justifications.

---

## Unit & Integration Testing

Run the test suite to verify data partitioning, prevalence preservation, metric calculations, and model interfaces:

```bash
pytest -v tests/
# or:
python run_tests.py -v tests/
```

All 13 unit tests validate:
- Zero data leakage between train/validation/test partitions.
- Exact prevalence preservation across random and duplicate-aware group splits.
- Proper threshold selection logic under target recall constraints.
- Tabular and PyTorch neural model initialization, forward pass, and predict_proba contracts.

---

## Final Course Submission Deliverables

- **Compiled Research Paper:** [`report/latex/main.pdf`](report/latex/main.pdf) (12 figures, comprehensive literature review, methodology, and empirical audit).
- **Presentation Deck:** [`presentation/FGGN_Cardiovascular_Research_Presentation_Final.pptx`](presentation/FGGN_Cardiovascular_Research_Presentation_Final.pptx) (11 validated widescreen slides with speaker notes).
- **Presentation Script & Outline:** [`presentation/slide_outline.md`](presentation/slide_outline.md).
- **Reconciled Predictions:** `artifacts/robustness/test_predictions.csv.gz` (951,300 per-sample test predictions reconciled with zero discrepancy).

---

## Important Disclaimers

1. **Self-Reported Survey Data:** CDC BRFSS data reflects self-reported telephonic interview responses regarding prior coronary heart disease or myocardial infarction, not direct clinical or laboratory measurements.
2. **Tabular Geometry:** CNN and GRU models impose an arbitrary 1D sequential ordering on non-sequential tabular features. Our findings confirm that tree-based gradient boosting models (HistGradientBoosting) are substantially better aligned with tabular survey data.
3. **Medical Disclaimer:** This work is strictly an educational research benchmark and must not be used for diagnosis, clinical triage, or patient treatment decisions.
