# Report archive

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
