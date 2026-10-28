# First prompt to paste into Antigravity

Open the root of this project in Antigravity. Use `/plan` and paste the following:

---

We are building a course research project based on the supplied IEEE Access 2024 paper on EnsCVDD-Net and BlCVDD-Net. Read `AGENTS.md`, `ANTIGRAVITY_IMPLEMENTATION_PLAN.md`, `PROJECT_PLAN.md`, `README.md`, and the existing source code before proposing changes.

I have the real CSV. First locate it in the workspace or ask me for its exact path if it is not present. It must be audited by target/schema, not guessed from filename. Do not use `synthetic_smoke.csv` as research data.

**Planning only at this stage; do not edit code or start long model runs yet.** Inspect the current repository and the data file, identify the precise gaps/bugs, and return an actionable implementation plan with phases, files to modify, tests, exact commands, compute estimates/risks, and acceptance criteria. Pay special attention to leakage-safe splitting, ADASYN within training folds only for cross-validation/OOF stacking, a naturally imbalanced untouched test set, PBCC fit on training labels only, and validation-only threshold selection. Check the current tabular stack for global resampling before internal OOF folds.

Use a 70/15/15 stratified train/validation/test split with seed 42 for the initial experiment. Compare majority, Logistic Regression, Random Forest, Extra Trees, Histogram Gradient Boosting, the paper-family LeNet/GRU/hybrid/blending models, and the proposed Feature-Group Gated Network. Treat +2 percentage points accuracy as a stretch target, not a guaranteed result. Primary metric is test PR-AUC/average precision; also report precision, recall, specificity, F1, balanced accuracy, accuracy, ROC-AUC, Brier score, confusion counts and runtime. Evaluate fixed threshold 0.50 and a validation-selected threshold that targets recall >= 0.80. Never use the test set for model selection.

Do a short related-work check before claiming novelty. Do not fabricate citations, metrics, successful tests, or exact reproduction of architecture details not supplied by the paper. Give me the plan first for review; after approval, implement incrementally and report actual test/command results at each checkpoint.

---

After Antigravity returns the plan, review/approve it before asking it to execute. The full requirements are in `ANTIGRAVITY_IMPLEMENTATION_PLAN.md` and persistent constraints are in `AGENTS.md`.
