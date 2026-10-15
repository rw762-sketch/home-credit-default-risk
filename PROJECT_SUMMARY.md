# Project overview

An application-table Home Credit credit-risk study with model comparison, development-selected cutoff, calibration, and distribution-drift diagnostics.

Start with the [README](README.md) for measured results, validation, reproduction, and limitations. Read the [technical report](reports/2026-10-04_03_home-credit/2026-10-04_02_report.md) for the detailed experimental narrative. The [experiment ledger](EXPERIMENTS.md) distinguishes completed comparisons from untested hypotheses, and [REPORTS.md](REPORTS.md) indexes the saved presentations.

The selected histogram gradient boosting model achieved **0.7605 mean development CV AUC** and **0.7654 reused-holdout AUC**. The development F2 cutoff is **0.0844**, with **67.19% holdout recall**, **17.69% precision**, and **15,517 false positives**. The threshold changes the error tradeoff; it does not improve ranking.

The project uses 307,511 labeled applications and produces 48,744 local competition-test probabilities. No leaderboard score, independent temporal validation or demonstrated lending benefit is reported.

The separate full-development history follow-up compares three feature sets for HGB and LightGBM on the same 246,008 development rows and five folds. `hgb_application_bureau_installments` has the highest mean fold AUC in this fixed comparison (0.773351), with OOF AP 0.260517. This is development evidence; the default application-only pipeline is unchanged and the holdout is excluded. See the [experiment ledger](EXPERIMENTS.md#full-development-staged-follow-up) for paired changes, precision/recall, exclusions and limitations.
