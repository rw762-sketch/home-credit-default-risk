# Experiment ledger

This ledger separates measured comparisons, descriptive diagnostics, and future hypotheses. Unless dated otherwise, results refer to the October 4, 2026 application-table experiment and monitoring extension.

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

1. Test previous-application history without the neutral application-feature bundle, and test other historical tables separately.
2. Assess whether gains persist under a bounded, prespecified tuning protocol; the fixed LightGBM comparison is measured below.
3. Evaluate fold-fitted recalibration and other stated cutoff objectives.
4. Freeze the selected pipeline and obtain defensible independent evaluation data.

These are proposals, not achieved gains. Measured history-feature and fixed-budget/inner-stopping comparisons are dated below. Broad hyperparameter search, stacking and leaderboard evaluation remain unreported.

## October 14, 2026: bounded bureau feature ablation

Question: do application-time bureau credit-history aggregates add predictive information when the model and validation rows are held fixed?

Protocol: recreate the original seed-42 80/20 split and five development folds, then select 30,000 development applications stratified by label and original fold. All selected applicant/fold assignments were checked against the saved October 4 development predictions. Both variants use the existing fold-fitted preprocessing and histogram boosting configuration (at most 150 iterations, 15 leaves, L2 regularization 1, seed 42), with two native threads. The original 61,503-row holdout is excluded and never scored. No tuning, threshold optimization, final refit, or submission is performed.

| Variant | Features | Mean fold ROC-AUC | Pooled OOF average precision | Pooled OOF Brier | Pooled OOF log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Application only | 123 | 0.736690 | 0.217979 | 0.068982 | 0.251978 |
| Application + bureau | 137 | 0.739088 | 0.224493 | 0.068760 | 0.251193 |

Mean paired fold AUC gain: **0.002399**, positive on all five folds. Fold AUC changes were +0.002867, +0.001584, +0.000775, +0.001388, and +0.005381. Average precision improved on four folds, with a small decline on one. These are descriptive development comparisons, not significance tests or independent replication. The capped baseline must not be compared directly with the full-data October 4 AUC.

Audit: the archive has 1,716,428 bureau records; 143,315 belong to the selected applications. One scoped record has a future update date and is excluded, leaving 143,314 eligible records. There are 25,662 applications with eligible history and 4,338 without it. Credit/update dates must be known and at or before application day 0; supplied future or invalid nonmissing actual closures are excluded. Scheduled future maturity dates are not used. Signed net debt is retained, missing balances remain missing, and ratios use only matching known debt/positive-credit loans. Snapshot filtering does not independently verify the provider's historical availability.

Decision: keep this small gain as an **opt-in research feature family**. The default application-only training command and published results remain unchanged. Full-development replication and fresh external/time-based evaluation remain future work.

Reproduce with `python -m src.compare_bureau --bureau data/archive/2026-10-04_00_home-credit-default-risk.zip --max-development-rows 30000 --folds 5 --threads 2`. Local Git-ignored comparison folders contain the report, exact assignments, per-fold metrics, OOF predictions, feature schema, exclusion audit, and input/code/version manifest. The tests cover join cardinality/order, missing histories, duplicate IDs, label isolation, numerical edges, application-time boundaries, matched folds, and output preservation.

## Full-development staged follow-up

Experiment date: October 15, 2026 (UTC).

Question: do the bureau gain, an additional installment-payment family, and a fixed LightGBM preset improve ranking and operating-point precision on the same full development set?

Protocol: 246,008 development applications, 19,860 positives (8.0729%), original seed-42 five folds. The 61,503-row holdout is excluded and never scored. The regenerated application-only HGB probabilities matched every original saved OOF probability within 1e-12, with identical IDs, labels and folds. The full application/bureau HGB ablation took 203.9 fit seconds; the staged runner verified and reused these results instead of fitting them again.

HGB retains the original at-most-150-iteration/15-leaf/L2=1 configuration and internal early stopping. LightGBM uses 200 trees, learning rate 0.05, 15 leaves, L2=1 and seed 42; deterministic column-wise training is requested. All fits use two native threads and the same fold-fitted preprocessing. This is a comparison of configured models, not equally tuned or proven best model families.

| Variant | Features | Mean fold AUC | Pooled OOF AP | OOF Brier | OOF log loss | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgb_application_only | 123 | 0.760478 | 0.242690 | 0.067821 | 0.245861 | 92.3 |
| hgb_application_bureau | 137 | 0.765823 | 0.251943 | 0.067409 | 0.244112 | 111.6 |
| hgb_application_bureau_installments | 153 | 0.773351 | 0.260517 | 0.066946 | 0.241846 | 125.5 |
| lgbm_application_only | 123 | 0.761839 | 0.244323 | 0.067741 | 0.245497 | 46.0 |
| lgbm_application_bureau | 137 | 0.766486 | 0.252620 | 0.067380 | 0.244011 | 50.1 |
| lgbm_application_bureau_installments | 153 | 0.773288 | 0.260372 | 0.066961 | 0.241984 | 55.2 |

Paired fold changes (candidate minus baseline):

| Baseline → candidate | Mean AUC change | Positive AUC folds | Mean AP change | Positive AP folds |
| --- | ---: | ---: | ---: | ---: |
| hgb_application_only → hgb_application_bureau | +0.005345 | 5/5 | +0.009278 | 5/5 |
| hgb_application_bureau → hgb_application_bureau_installments | +0.007528 | 5/5 | +0.008482 | 5/5 |
| lgbm_application_only → lgbm_application_bureau | +0.004647 | 5/5 | +0.008453 | 5/5 |
| lgbm_application_bureau → lgbm_application_bureau_installments | +0.006802 | 5/5 | +0.007548 | 5/5 |
| hgb_application_only → lgbm_application_only | +0.001361 | 5/5 | +0.001534 | 4/5 |
| hgb_application_bureau → lgbm_application_bureau | +0.000663 | 5/5 | +0.000710 | 4/5 |
| hgb_application_bureau_installments → lgbm_application_bureau_installments | -0.000063 | 1/5 | -0.000224 | 2/5 |

Pooled OOF capacity tradeoffs; top-k policies use a deterministic applicant-ID tie break:

| Variant | Highest-risk fraction | Flagged | TP | FP | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hgb_application_only | top_5% | 12,300 | 4,064 | 8,236 | 33.04% | 20.46% |
| hgb_application_only | top_10% | 24,601 | 6,641 | 17,960 | 26.99% | 33.44% |
| hgb_application_only | top_20% | 49,202 | 10,365 | 38,837 | 21.07% | 52.19% |
| hgb_application_bureau | top_5% | 12,300 | 4,191 | 8,109 | 34.07% | 21.10% |
| hgb_application_bureau | top_10% | 24,601 | 6,779 | 17,822 | 27.56% | 34.13% |
| hgb_application_bureau | top_20% | 49,202 | 10,566 | 38,636 | 21.47% | 53.20% |
| hgb_application_bureau_installments | top_5% | 12,300 | 4,283 | 8,017 | 34.82% | 21.57% |
| hgb_application_bureau_installments | top_10% | 24,601 | 6,924 | 17,677 | 28.15% | 34.86% |
| hgb_application_bureau_installments | top_20% | 49,202 | 10,799 | 38,403 | 21.95% | 54.38% |
| lgbm_application_only | top_5% | 12,300 | 4,070 | 8,230 | 33.09% | 20.49% |
| lgbm_application_only | top_10% | 24,601 | 6,636 | 17,965 | 26.97% | 33.41% |
| lgbm_application_only | top_20% | 49,202 | 10,388 | 38,814 | 21.11% | 52.31% |
| lgbm_application_bureau | top_5% | 12,300 | 4,202 | 8,098 | 34.16% | 21.16% |
| lgbm_application_bureau | top_10% | 24,601 | 6,791 | 17,810 | 27.60% | 34.19% |
| lgbm_application_bureau | top_20% | 49,202 | 10,527 | 38,675 | 21.40% | 53.01% |
| lgbm_application_bureau_installments | top_5% | 12,300 | 4,303 | 7,997 | 34.98% | 21.67% |
| lgbm_application_bureau_installments | top_10% | 24,601 | 6,951 | 17,650 | 28.25% | 35.00% |
| lgbm_application_bureau_installments | top_20% | 49,202 | 10,791 | 38,411 | 21.93% | 54.34% |

Installment audit: 13,605,401 source payment rows, 9,279,760 scoped development rows, 2,091 time/key exclusions, 117,688 ambiguous/duplicate exclusions, 9,159,981 retained events, 8,706,315 obligations, 12,704 applicants without eligible history.

Observed payment tranches are consolidated before counting scheduled amounts. Both due and payment dates must be known and <= application day 0. Future payment rows do not affect amounts, schedule versions or ambiguity decisions. Conflicting schedules/versions and exact duplicate event groups are conservatively excluded. Shortfalls require all contributing amounts to be known; ratios handle zero denominators. These features describe observed past payments and do not enumerate every unpaid obligation or prove source snapshot availability.

The highest mean-AUC configuration in this fixed comparison is `hgb_application_bureau_installments` (0.773351); its top-5% precision is 34.82% with 21.57% recall. This choice is based on development evidence and is not promoted to the default application-only training pipeline. Inspect all operating points; a better AUC/AP does not guarantee better precision at every capacity. Fold variation is descriptive, not a significance test. Independent final/temporal validation, calibration experiments and profitability remain unresolved.

LightGBM with both history families was slightly weaker than HGB on mean AUC (0.773288 versus 0.773351) and pooled AP (0.260372 versus 0.260517), while its five fits took 55.2 versus 125.5 seconds. At a fixed top-5% capacity, LightGBM caught 4,303 positives versus HGB's 4,283 (34.98% versus 34.82% precision); at top 10% it caught 27 more positives, while at top 20% it caught 8 fewer. Model choice therefore depends on the operating point and runtime as well as overall ranking. Even the top-5% LightGBM group contains 7,997 false positives among 12,300 flags; these gains do not establish high-precision lending decisions.

The staged process took 299.3 wall seconds, including input verification/aggregation and new fits; table fit times include the reused HGB fits. The runner checkpoints every fold atomically and verifies resume state before skipping fits. Publication review additionally checks exact variant/fold coverage and recomputes per-fold metrics before accepting cached results; these validation-only changes do not alter the measured features or model configurations.

Artifacts: local Git-ignored `outputs/2026-10-14_03_bureau-full-development/` and `outputs/2026-10-15_01_history-full-development/`. They retain exact folds, OOF predictions, per-fold and pooled metrics, top-5/10/20% and 0.5 counts, paired feature/algorithm deltas, input/code hashes, versions, audit policies and runtime. See the README for commands and the process-local OpenMP fallback used by the already-installed LightGBM.

Published compact evidence for the full-development follow-up is in the [tracked report](reports/2026-10-15_01_history-development/2026-10-15_02_report.md); row-level outputs remain local.

## November 5, 2026: validation improvements and saved-score audit

The proposed `uncapped-history-stages` branch at `9a38fb3` was reviewed before integration. Its new feature families and larger budgets are experiments, not established gains. The review fixed missing artifact-name registrations, isolated inner stopping preprocessing for both algorithms, and added full outer-training refits at the selected iteration count. The legacy no-patience path remains available. HGB selection excludes its patience tail; recorded actual refit iterations, selected length and search rounds are separate. Changed feature/stopping code invalidates checkpoints; changed HGB settings reject the original bureau cache.

Application extensions bundle external-score summaries **and** employment/age, goods/credit and per-family-member income ratios. Any stage gain belongs to that bundle; it cannot be attributed to external-score summaries alone. Previous-application amounts are nominal approved amounts, not current debt obligations. Decision dates are screened at day 0; source availability still requires independent provenance review.

A separate read of the source decision-day column found **zero day-zero decisions in all 1,670,214 previous applications**, and zero future decisions scoped to the 246,008 development IDs. Therefore changing the decision boundary from ≤0 to <0 cannot change this dataset's previous-application aggregates. This empirical equivalence does not establish within-day availability in a different dataset. [Aggregate coverage audit](reports/2026-11-05_03_original-history-audit/2026-11-05_43_previous-audit.json).

An aggregate-only [saved-score audit](reports/2026-11-05_03_original-history-audit/2026-11-05_02_report.md) recomputed the October 15 per-fold/pooled metrics after checking prediction/cohort hashes and exact assignments. With 1,000 paired applicant resamples within each fold/label stratum (seed 42), application-only HGB versus HGB with bureau/installments has pooled OOF AUC change **+0.012903**, conditional percentile interval **[+0.011512, +0.014155]**; AP change **+0.017827**, interval **[+0.015098, +0.020646]**. Fixed-width calibration tables, Brier/log loss and source/code/version manifests accompany it.

These are conditional resampling diagnostics on already inspected fitted scores, **not generalization confidence intervals or significance tests**. They exclude refitting, overlapping-training-fold dependence, selection and population shift. Pooled OOF AUC is distinct from mean fold AUC, the model-selection metric. No new performance gain, fresh validation, calibrator, tuned cutoff, final refit or submission is implied by this audit.

The exact-package snapshot is `requirements-experiment.txt`; `requirements.txt` remains the bounded installation interface. Local software checks include the LightGBM runtime, future-record exclusion, inner-preprocessor isolation/full-fold refit, checkpoint boundaries, exact ties, deterministic paired resampling, and weighted-score agreement with explicit scikit-learn resampling. The full higher-budget five-stage development experiment is recorded separately after completion; the original holdout is excluded.

## Evidence links

- [Model comparison table](reports/2026-10-04_03_home-credit/data/2026-10-04_06_model-comparison.csv)
- [Cutoff comparison table](reports/2026-10-04_03_home-credit/data/2026-10-04_24_threshold-comparison.csv)
- [Calibration table](reports/2026-10-04_03_home-credit/data/2026-10-04_26_calibration.csv)
- [Drift table](reports/2026-10-04_03_home-credit/data/2026-10-04_28_drift.csv)
- [Full methodology and interpretation](reports/2026-10-04_03_home-credit/2026-10-04_02_report.md)

## November 5: measured higher-budget staged comparison

The November 5 comparison reaches **0.780062 mean fold AUC for HGB and 0.780078 for LightGBM** with the tested history families: effectively tied ranking results in this experiment. LightGBM took 288.5 fit seconds versus HGB's 1,082.3; HGB had slightly higher pooled OOF AP (0.270274 versus 0.269765). Previous-application history improved both algorithms, while the application-feature bundle showed no clear gain. These are previously inspected development results; the holdout remains excluded and independent final evaluation is unresolved.

[Full results and conditional audits](reports/2026-11-05_04_extended-history/2026-11-05_02_report.md).

The application-feature bundle changed mean fold AUC by -0.000229 (HGB) and -0.000523 (LightGBM); its conditional pooled AUC/AP intervals all include zero. Adding previous-application history then changed mean fold AUC by +0.003811 and +0.003869, respectively, positive on all five folds for each algorithm. The paired pooled AUC intervals were [+0.002981, +0.004634] for HGB and [+0.003105, +0.004687] for LightGBM. These fixed-score resampling intervals do not establish generalization or statistical significance.

Decision: retain previous-application history as a research candidate; do not claim the application-feature bundle improved performance. Test previous history without that bundle before choosing the simplest final feature set. The algorithms' tiny ranking difference does not support a clear winner: LightGBM is faster and marginally better at top-5% capacity, while HGB has higher pooled AP and top-10% precision here. All 50 searches stopped below 1,000 rounds; the full run took 6,018.7 seconds, with the original holdout excluded.
