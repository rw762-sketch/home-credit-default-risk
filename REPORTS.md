# Report archive

## Higher-budget staged development comparison (November 5)

[Report](reports/2026-11-05_04_extended-history/2026-11-05_02_report.md) · [Aggregate tables](reports/2026-11-05_04_extended-history/data/)

The November 5 comparison reaches **0.780062 mean fold AUC for HGB and 0.780078 for LightGBM** with the tested history families: effectively tied ranking results in this experiment. LightGBM took 288.5 fit seconds versus HGB's 1,082.3; HGB had slightly higher pooled OOF AP (0.270274 versus 0.269765). Previous-application history improved both algorithms, while the application-feature bundle showed no clear gain. These are previously inspected development results; the holdout remains excluded and independent final evaluation is unresolved.

## Saved development score audit (November 5)

[Report](reports/2026-11-05_03_original-history-audit/2026-11-05_02_report.md) · [Conditional paired intervals](reports/2026-11-05_03_original-history-audit/2026-11-05_44_bootstrap-deltas.csv) · [Calibration bins](reports/2026-11-05_03_original-history-audit/2026-11-05_26_calibration.csv) · [Manifest](reports/2026-11-05_03_original-history-audit/2026-11-05_04_manifest.json)

Re-audits the October 15 saved predictions without training. The paired intervals condition on fixed OOF scores and preserve fold/class sizes; they are not independent validation or generalization confidence intervals. Only aggregate tables are tracked.

## History features and fixed-model comparison (development only)

[Report](reports/2026-10-15_01_history-development/2026-10-15_02_report.md) · [Compact tables](reports/2026-10-15_01_history-development/data/) · [Manifest](reports/2026-10-15_01_history-development/2026-10-15_04_manifest.json)

The October 15 follow-up compares application, bureau and installment features for HGB and LightGBM on the same full development folds. It reports ranking, probability quality, top-5/10/20% precision/recall, paired changes and runtime. The holdout is not scored. Applicant-level outputs remain local.

## Featured: model comparison, cutoff, calibration, and drift

[Markdown](reports/2026-10-04_03_home-credit/2026-10-04_02_report.md) · [PDF](reports/2026-10-04_03_home-credit/2026-10-04_03_results.pdf) · [HTML](reports/2026-10-04_03_home-credit/2026-10-04_01_index.html) · [Start page](reports/2026-10-04_03_home-credit/2026-10-04_00_start-here.md)

The October 4, 2026 monitoring run compares three classifiers and selects the F2 cutoff using development OOF predictions. It adds calibration, distribution drift, data-quality, and age summaries. Follow-up holdout results reuse inspected data.

- [Compact tables](reports/2026-10-04_03_home-credit/data/)
- [Figures](reports/2026-10-04_03_home-credit/figures/)
- [Source artifact manifest](reports/2026-10-04_03_home-credit/2026-10-04_04_manifest.json)
- [Experiment ledger](EXPERIMENTS.md)

## Initial application-table benchmark

[Markdown](reports/2026-10-04_01_home-credit/2026-10-04_02_report.md) · [PDF](reports/2026-10-04_01_home-credit/2026-10-04_03_results.pdf)

The original benchmark reports model comparison, the fixed 0.5 cutoff, and SHAP. It predates the monitoring extension. Both experiments use the same model configuration, seed, and split; the repeated AUC is not independent replication.

## Local-only artifacts

Full row-level predictions, fitted models, training logs, synthetic smoke checks, and downloaded data remain local and are excluded from Git. The tracked report figures and compact tables are sufficient to inspect the reported results. Source hashes identify the local inputs used by the report builder; they do not replace the underlying files.

## Regenerate

```bash
python -m pip install -r requirements.txt -r requirements-reporting.txt
python clean_report.py
```

Reporting reads the latest completed real-data output without retraining. To select an existing local run, pass `--source outputs/<run-folder>`. New runs receive separate dated, numbered folders; presentation regeneration updates the corresponding report.
