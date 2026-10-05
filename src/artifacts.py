"""Consistent dated, numbered filenames for saved experiments and reports."""
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

TZ = ZoneInfo('America/New_York')
ORDER = {
    'START_HERE.md': 0, 'index.html': 1, 'report.md': 2,
    'results.pdf': 3, 'manifest.json': 4, 'metrics.json': 5,
    'model_comparison.csv': 6, 'model_comparison.png': 7,
    'roc_curve.png': 8, 'precision_recall_curve.png': 9,
    'confusion_matrix.png': 10, 'shap_summary.png': 11,
    'feature_importance.csv': 12, 'feature_importance.png': 13,
    'missing_values.csv': 14, 'target_distribution.png': 15,
    'correlation_heatmap.png': 16, 'holdout_predictions.csv': 17,
    'evaluation_model.joblib': 18, 'submission.csv': 19,
    'submission_model.joblib': 20, 'training.log': 21,
    'development_predictions.csv': 22, 'threshold_sweep.csv': 23,
    'threshold_comparison.csv': 24, 'threshold_curve.png': 25,
    'calibration.csv': 26, 'calibration_curve.png': 27,
    'drift.csv': 28, 'drift_chart.png': 29,
    'monitoring_metrics.json': 30, 'data_quality.csv': 31,
    'age_risk.csv': 32, 'age_risk.png': 33,
}


def run_date(folder):
    """Read the experiment date from its directory, also for figures/data children."""
    for part in reversed(Path(folder).parts):
        if re.match(r'^\d{4}-\d{2}-\d{2}_', part):
            return part[:10]
    return datetime.now(TZ).strftime('%Y-%m-%d')


def artifact_path(folder, name):
    """Resolve a canonical artifact name to its dated, ordered filename."""
    path = Path(name)
    number = ORDER[path.name]
    slug = path.stem.lower().replace('_', '-')
    return Path(folder) / f'{run_date(folder)}_{number:02d}_{slug}{path.suffix}'


def new_run(kind, root=Path('outputs')):
    day = datetime.now(TZ).strftime('%Y-%m-%d')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = root / f'{day}_{number:02d}_{kind}'
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


def latest_run(root, kind='home-credit'):
    runs = sorted(p for p in Path(root).glob(f'????-??-??_*_{kind}')
                  if p.is_dir() and artifact_path(p, 'metrics.json').exists()
                  and artifact_path(p, 'report.md').exists())
    if not runs:
        raise FileNotFoundError(f'No saved {kind} run found in {root}; run src.train first.')
    return runs[-1]
