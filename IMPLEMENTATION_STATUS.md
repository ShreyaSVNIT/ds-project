# Implementation Status

Update this file after each milestone. Do not mark a task complete without a command result or saved artifact to support it.

## Current state on handoff
- [x] Starter repository and paper-family model scaffold exist.
- [x] Project-level Antigravity instructions and phased implementation specification added.
- [x] Locate and validate the user's real CSV in the Antigravity workspace.
- [x] Inspect current code and identify/fix methodology or implementation issues.
- [x] Run data audit and save `artifacts/data_audit/` outputs.
- [x] Run tests and quick real-data tabular baseline.
- [x] Run paper-family neural models on real data.
- [x] Run proposed FGGN and ablation study.
- [x] Audit evidence and trace FGGN discrepancy between ADASYN vs class-weighting runs.
- [x] Run 5-seed study across seeds 42 (exploratory), 101, 2024, 777, 9999 (confirmatory).
- [x] Persist 951,300 individual per-example test predictions to `artifacts/robustness/test_predictions.csv.gz`.
- [x] Execute metric reconciliation script with zero discrepancy against reported values.
- [x] Probability calibration evaluation: constant-prevalence baseline, Platt scaling, and Isotonic regression on validation only.
- [x] Calibration slope and intercept evaluation under Van Calster clinical hierarchy (`artifacts/calibration/mean_calibration_summary.csv`).
- [x] Generate reliability diagrams across all five models with slope/intercept annotations (`artifacts/calibration/reliability_diagrams.png`).
- [x] Paired statistical comparisons with 95% confidence intervals and Nadeau-Bengio corrected repeated-holdout variance (`artifacts/calibration/statistical_paired_summary.csv`).
- [x] Subgroup robustness and prevalence sensitivity analysis across Age, Sex, and Income (`artifacts/subgroups/`).
- [x] Literature review and related work matrix updated with peer-reviewed DOIs and novelty assessment (`research/related_work_matrix.md`).
- [x] Full test suite expanded and passing (13/13 unit tests passed).
- [x] Update paper reproduction notes, report outline, and presentation slide deck outline.

## Experiment log

| Date | Phase | Command | Status | Evidence / notes |
|---|---|---|---|---|
| 2026-10-17 | Phase 1 (Data & Audit) | `python3 scripts/audit_dataset.py --data data/raw/heart_disease_health_indicators_BRFSS2015.csv --out artifacts/data_audit` | DONE | SHA256 verified, N=253,680, 22 cols, prevalence=0.094186, 0 missing, 23,899 duplicates. Saved audit JSON/CSVs/plot and split indices. |
| 2026-10-17 | Phase 1 (Testing) | `pytest -v tests/test_data_pipeline.py` | DONE | 5/5 unit tests passed (disjointness, prevalence, schema contract, PBCC isolation, threshold). |
| 2026-10-17 | Phase 2 (Full Tabular) | `python3 scripts/run_tabular_experiments.py --out artifacts/tabular_full --resampling none --top-k 21` | DONE | HistGradientBoosting top PR-AUC 0.3732 / ROC-AUC 0.8485; Stacked Blender PR-AUC 0.3653; 80.0% recall target achieved on held-out test data. |
| 2026-10-17 | Phase 3 (Full Neural) | `python3 scripts/run_paper_models.py --epochs 10 --batch-size 1024 --out artifacts/neural_full` | DONE | Ran with ADASYN: LeNet1D PR-AUC 0.3415; GRU PR-AUC 0.3215; EnsCVDD PR-AUC 0.3311; FGGN PR-AUC 0.3212. Checkpoint saved to `artifacts/neural_full/models/`. |
| 2026-10-17 | Phase 4 (Ablations) | `python3 scripts/run_ablation_study.py --out artifacts/ablations --epochs 5` | DONE | FGGN with class-weighted loss (no resampling) achieved PR-AUC 0.3654 / ROC-AUC 0.8467, while ADASYN dropped PR-AUC to 0.3268. |
| 2026-10-18 | Phase A (Audit & Trace) | `python3 -c "recompute and trace FGGN checkpoints"` | DONE | Resolved discrepancy: 0.3654 came from class-weighted ablation run, while `artifacts/neural_full/models/` stored the ADASYN checkpoint (0.3212). Both confirmed reproducible. |
| 2026-10-18 | Phase C (5-Seed Study) | `python3 scripts/run_multiseed_robustness.py --seeds 42 101 2024 777 9999 --epochs 5 --out artifacts/robustness` | DONE | 5-seed study complete: HistGradientBoosting PR-AUC $0.3717 \pm 0.0067$; UniformGroupNet $0.3693 \pm 0.0070$; FGGN $0.3689 \pm 0.0068$; LogisticRegression $0.3673 \pm 0.0063$; EnsCVDD-Net $0.3563 \pm 0.0055$. |
| 2026-10-18 | Phase D (Reconciliation) | `python3 scripts/verify_and_reconcile_metrics.py --predictions artifacts/robustness/test_predictions.csv.gz --summary artifacts/robustness/all_seeds_results.csv` | DONE | Recomputed all metrics across 951,300 test predictions; confirmed 0.0000000000 discrepancy against summary tables. |
| 2026-10-18 | Final Validation (Calibration & Stats) | `python3 scripts/evaluate_calibration_and_stats.py --out artifacts/calibration` | DONE | Evaluated Platt scaling and Isotonic regression on validation only; reduced Brier score from 0.1654 to 0.0703 (beating baseline 0.0853). Computed calibration slope/intercept and Nadeau-Bengio corrected resampled CIs. |
| 2026-10-18 | Final Validation (Subgroups) | `python3 scripts/evaluate_subgroups.py --out artifacts/subgroups` | DONE | Evaluated demographic slices across Age, Sex, and Income. Generated per-seed and mean summary tables, plus publication figure `subgroup_metrics_comparison.png`. |
| 2026-10-18 | Full Test Suite | `pytest -v` | DONE | 13/13 unit and integration tests passed in 4.97s. |

## Decisions and deviations

- **Resolution of FGGN Discrepancy:** The earlier research report quoted the class-weighted FGGN score from the ablation matrix (PR-AUC 0.3654), whereas the checkpoint stored in `artifacts/neural_full/models/` was generated under the ADASYN run (PR-AUC 0.3212). Under the frozen 5-seed class-weighted protocol, FGGN achieves $0.3689 \pm 0.0068$, confirming class-weighted performance.
- **Model Hierarchy & Positioning:** HistGradientBoosting is documented as the top-ranking model. FGGN and UniformGroupNet perform identically within standard deviation ($0.3689$ vs $0.3693$). Learned gating is reported as an investigational feature attribution mechanism that adds complexity without ranking gain over uniform group pooling.
- **Prevalence Consistency Framing:** The 91.7% accuracy / 96% precision claim in Khan et al. (IEEE Access 2024) is documented as a mathematical inconsistency with natural 9.42% prevalence, characteristic of evaluation on an artificially balanced test set, without asserting deliberate falsification.
- **Prediction Persistence:** Persisted 951,300 individual per-example test predictions (`artifacts/robustness/test_predictions.csv.gz`) with verified zero-discrepancy reconciliation.
- **Probability Calibration Protocol:** Calibrators (Platt and Isotonic) were fitted exclusively on validation predictions; test labels were never accessed during calibrator fitting.
- **Statistical Interpretation:** Applied Nadeau & Bengio (2003) corrected resampled variance ($\hat{\sigma}^2_{\text{corr}} = (1/K + n_{\text{test}}/n_{\text{train}})\hat{S}^2$) to account for overlapping training data across repeated holdout splits.
- **Subgroup Analysis Framing:** Performance disparities across demographic subgroups are documented as prevalence-driven screening dynamics, explicitly noting that these analyses do not constitute proof of algorithmic fairness.

## Paper expansion status
- [x] Expanded `report/latex/main.tex` with the seven requested literature subsections, research-gap table, dataset contract, architecture formulations, leakage-safe protocol, measured evaluation subsections, discussion, limitations, and future work.
- [x] Added 15 bibliography entries covering cardiovascular prediction, tabular neural architectures, oversampling, calibration, statistical testing, and reporting guidance.
- [x] Added persisted reliability, subgroup, target-distribution, and SHAP figures to `report/latex/figures/`; retained the existing comparison, prevalence, resampling, gating, threshold, and decision-curve figures in the paper.
- [x] Corrected the prevalence audit language to distinguish an arithmetic implication from proof of the original split or resampling procedure.
- [x] Full LaTeX compilation completed with Tectonic; warnings remain for underfull/overfull boxes and the legacy `algorithmic.sty` encoding.
- [x] Duplicate-group-aware splitting and a 70/80/85/90/95 percent recall threshold grid completed as separate evaluation artifacts.
- [x] Installed Tectonic and compiled `report/latex/main.tex` successfully; warnings remain for underfull/overfull boxes and the legacy `algorithmic.sty` encoding.
- [x] Completed five-seed validation-selected threshold sweep at 70, 80, 85, 90, and 95 percent recall targets (`artifacts/robustness_threshold_grid/`).
- [x] Completed separate five-seed exact duplicate-predictor-group-aware evaluation (`artifacts/robustness_duplicate_group/`) and split prevalence/disjointness audit.
- [x] Audited and separated the exploratory FGGN-family benchmark from the primary confirmatory table; documented its different optimizer/epoch budget and broader model search.
- [x] Reconciled hybrid-ensemble metadata against `multiseed_blend_summary.csv`; the highest observed fixed-weight scan is 70/30 HGB/FGGN at PR-AUC 0.3730 +/- 0.0063, while the 80/20 blend is 0.3728 +/- 0.0064.
- [x] Corrected the stale single-seed hybrid summary language so it is no longer presented as multi-seed evidence.
