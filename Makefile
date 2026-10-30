# Cardiovascular Classification Research Makefile
# Reproducibility automation for Khan et al. (2024) audit and FGGN extension

PYTHON := python3
DATA := data/raw/heart_disease_health_indicators_BRFSS2015.csv

.PHONY: help all audit tabular neural ablations robustness calibration subgroups figures test clean

help:
	@echo "Available commands:"
	@echo "  make audit         - Run dataset cryptographic audit and schema verification"
	@echo "  make tabular       - Run leakage-safe tabular baselines"
	@echo "  make neural        - Run paper-family neural architectures and FGGN"
	@echo "  make ablations     - Run architectural ablation study (gating, resampling)"
	@echo "  make robustness    - Run 5-seed confirmatory study"
	@echo "  make calibration   - Run Platt scaling and Isotonic regression evaluation"
	@echo "  make subgroups     - Run demographic fairness and slice analysis"
	@echo "  make dca           - Run clinical Decision Curve Analysis"
	@echo "  make shap          - Run SHAP feature attribution and agreement suite"
	@echo "  make blend         - Run HistGBT + FGGN multi-seed probability blend"
	@echo "  make test          - Run full pytest unit test suite"
	@echo "  make all           - Execute complete end-to-end research pipeline"

audit:
	$(PYTHON) scripts/audit_dataset.py --data $(DATA) --out artifacts/data_audit
	$(PYTHON) scripts/verify_prevalence_inconsistency.py

tabular:
	$(PYTHON) scripts/run_tabular_experiments.py --out artifacts/tabular_full --resampling none --top-k 21

neural:
	$(PYTHON) scripts/run_paper_models.py --epochs 10 --batch-size 1024 --out artifacts/neural_full

ablations:
	$(PYTHON) scripts/run_ablation_study.py --out artifacts/ablations --epochs 5

robustness:
	$(PYTHON) scripts/run_multiseed_robustness.py --seeds 42 101 2024 777 9999 --epochs 5 --out artifacts/robustness
	$(PYTHON) scripts/verify_and_reconcile_metrics.py --predictions artifacts/robustness/test_predictions.csv.gz --summary artifacts/robustness/all_seeds_results.csv

calibration:
	$(PYTHON) scripts/evaluate_calibration_and_stats.py --out artifacts/calibration

subgroups:
	$(PYTHON) scripts/evaluate_subgroups.py --out artifacts/subgroups

dca:
	$(PYTHON) scripts/evaluate_dca.py

shap:
	$(PYTHON) scripts/evaluate_shap_agreement.py

blend:
	$(PYTHON) scripts/evaluate_multiseed_blend.py

test:
	pytest -v tests/

all: audit tabular neural ablations robustness calibration subgroups dca shap blend test
	@echo "All research milestones executed cleanly and verified."

