# Experiment ledger

This ledger separates measured comparisons, descriptive diagnostics, and future hypotheses. Results refer to the October 4, 2026 application-table experiment and monitoring extension.

## Measured comparisons

| Experiment | Hypothesis / objective | Evidence | Decision |
| --- | --- | --- | --- |
| Model families | Nonlinear models may improve ranking over logistic regression | Mean development CV AUC: histogram boosting 0.7605; logistic regression 0.7459; random forest 0.7373 | Retain histogram boosting for diagnostics |
| Classification cutoff | A recall-weighted objective should catch more positive cases | Development OOF F2 selects 0.0844; reused-holdout recall 67.19%, precision 17.69%, 15,517 false positives | Report the tradeoff; do not claim an operational policy |

All candidates share five development folds and the same features. The model-family comparison evaluates configured models, not each family's best possible configuration. The cutoff experiment reuses an inspected holdout and does not improve ROC-AUC.

## Descriptive diagnostics

| Diagnostic | Observation | What it does not establish |
| --- | --- | --- |
| Calibration | Reused-holdout Brier 0.06733; log loss 0.24398; decile calibration plot | Future calibration, fitted recalibration, or financial value |
| Distribution drift | Same-model score PSI 0.00384; raw bureau-enquiry and contract-type distributions differ | Test AUC or calibration; test labels are unavailable |
| Interpretation | External scores are prominent in the small SHAP sample | Causality or complete population-wide explanations |
| Data quality / age | Development-only missingness, sentinel counts, and age summaries saved | A tested feature improvement or causal age effect |

## Untested hypotheses

1. Add one applicant-linked history feature family and compare on identical development folds.
2. Evaluate LightGBM on the current features before adding model complexity.
3. Evaluate fold-fitted recalibration and other stated cutoff objectives.
4. Freeze the selected pipeline and obtain defensible independent evaluation data.

These are proposals, not achieved gains. No historical-feature ablation, LightGBM run, hyperparameter search, stacking, or leaderboard evaluation is reported.

## Evidence

- [Model comparison table](reports/2026-10-04_03_home-credit/data/2026-10-04_06_model-comparison.csv)
- [Cutoff comparison table](reports/2026-10-04_03_home-credit/data/2026-10-04_24_threshold-comparison.csv)
- [Calibration table](reports/2026-10-04_03_home-credit/data/2026-10-04_26_calibration.csv)
- [Drift table](reports/2026-10-04_03_home-credit/data/2026-10-04_28_drift.csv)
- [Full methodology and interpretation](reports/2026-10-04_03_home-credit/2026-10-04_02_report.md)
