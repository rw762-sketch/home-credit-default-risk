# Project overview

A Home Credit application and historical-feature credit-risk study with model comparison, development-selected cutoff, calibration, and distribution-drift diagnostics.

Start with the [README](README.md) for measured results, validation, reproduction, and limitations. Read the [technical report](reports/2026-10-04_03_home-credit/2026-10-04_02_report.md) for the detailed experimental narrative. The [experiment ledger](EXPERIMENTS.md) distinguishes completed comparisons from untested hypotheses, and [REPORTS.md](REPORTS.md) indexes the saved presentations.

The selected histogram gradient boosting model achieved **0.7605 mean development CV AUC** and **0.7654 reused-holdout AUC**. The development F2 cutoff is **0.0844**, with **67.19% holdout recall**, **17.69% precision**, and **15,517 false positives**. The threshold changes the error tradeoff; it does not improve ranking.

The project uses 307,511 labeled applications and produces 48,744 local competition-test probabilities. No leaderboard score, independent temporal validation or demonstrated lending benefit is reported.

The October 15 full-development history follow-up compares three feature sets for HGB and LightGBM on the same 246,008 development rows and five folds. `hgb_application_bureau_installments` has the highest mean fold AUC in this fixed comparison (0.773351), with OOF AP 0.260517. This is development evidence; the default application-only pipeline is unchanged and the holdout is excluded. See the [experiment ledger](EXPERIMENTS.md#full-development-staged-follow-up) for paired changes, precision/recall, exclusions and limitations.

## Higher-budget development follow-up

The November 5 comparison reaches **0.780062 mean fold AUC for HGB and 0.780078 for LightGBM** with the tested history families: effectively tied ranking results in this experiment. LightGBM took 288.5 fit seconds versus HGB's 1,082.3; HGB had slightly higher pooled OOF AP (0.270274 versus 0.269765). Previous-application history improved both algorithms, while the application-feature bundle showed no clear gain. These are previously inspected development results; the holdout remains excluded and independent final evaluation is unresolved.

[Full results and conditional audits](reports/2026-11-05_04_extended-history/2026-11-05_02_report.md).
