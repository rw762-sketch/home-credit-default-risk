"""Home Credit application-table benchmark. Run: python -m src.train --help."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.artifacts import artifact_path, new_run, TZ
from datetime import datetime

SEED = 42


def features(frame):
    """Only row-local transformations; no learned statistics or target access."""
    x = frame.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore').copy()
    if 'DAYS_EMPLOYED' in x:
        x['DAYS_EMPLOYED'] = x['DAYS_EMPLOYED'].replace(365243, np.nan)
    for name, numerator, denominator in [
        ('CREDIT_INCOME_RATIO', 'AMT_CREDIT', 'AMT_INCOME_TOTAL'),
        ('ANNUITY_INCOME_RATIO', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
        ('CREDIT_ANNUITY_RATIO', 'AMT_CREDIT', 'AMT_ANNUITY')]:
        if numerator in x and denominator in x:
            x[name] = x[numerator] / x[denominator].replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def validate(frame, folds):
    for column in ['TARGET', 'SK_ID_CURR']:
        if column not in frame:
            raise ValueError(f'Missing required column: {column}')
    if frame.SK_ID_CURR.isna().any() or frame.SK_ID_CURR.duplicated().any():
        raise ValueError('Applicant IDs must be nonmissing and unique.')
    if frame.TARGET.isna().any() or set(frame.TARGET.unique()) != {0, 1}:
        raise ValueError('TARGET must contain both 0 and 1, with no missing labels.')
    if folds < 2 or frame.TARGET.value_counts().min() < folds + 2:
        raise ValueError('Need at least two folds and enough examples of both classes.')


def pipeline(x, estimator):
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in x if c not in numeric]
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                     min_frequency=20))]), categorical)],
        verbose_feature_names_out=False)
    return Pipeline([('preprocess', preprocessor), ('model', estimator)])


def synthetic():
    """Generated test fixture, never a substitute for Home Credit results."""
    rng = np.random.default_rng(SEED)
    n = 1000
    income = rng.lognormal(11, .6, n)
    credit = rng.lognormal(12, .5, n)
    score = rng.uniform(0, 1, n)
    probability = 1 / (1 + np.exp(-(-3 + 3 * (1-score) + .1 * credit/income)))
    frame = pd.DataFrame({'SK_ID_CURR': np.arange(n), 'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit, 'AMT_ANNUITY': credit/15, 'EXT_SOURCE_2': score,
        'DAYS_EMPLOYED': rng.choice([-100, -1000, 365243], n),
        'NAME_INCOME_TYPE': rng.choice(['Working', 'Pensioner', 'Student'], n),
        'TARGET': rng.binomial(1, probability)})
    frame.loc[rng.choice(n, 100, replace=False), 'EXT_SOURCE_2'] = np.nan
    return frame


def saveplot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def run(args):
    if not args.demo and not args.train.exists():
        raise FileNotFoundError(f'Place application_train.csv at {args.train}, or use --demo.')
    frame = synthetic() if args.demo else pd.read_csv(args.train)
    validate(frame, args.folds)
    out = args.output or new_run('demo' if args.demo else 'home-credit')
    out.mkdir(parents=True, exist_ok=True)
    x, y = features(frame), frame.TARGET.astype(int)
    dev, hold = train_test_split(np.arange(len(frame)), test_size=.2,
                                stratify=y, random_state=SEED)
    xd, yd = x.iloc[dev], y.iloc[dev]
    xh, yh = x.iloc[hold], y.iloc[hold]
    if yd.value_counts().min() < args.folds:
        raise ValueError('Development split has too few minority samples for the requested folds.')
    # Descriptive outputs use development rows only, preserving the holdout.
    xd.isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(artifact_path(out, 'missing_values.csv'))
    yd.value_counts().sort_index().plot.bar(title='Development target distribution')
    plt.xlabel('TARGET (1 = repayment difficulty)')
    plt.ylabel('Applicants')
    saveplot(artifact_path(out, 'target_distribution.png'))
    selected = xd.select_dtypes(include='number').columns[:12]
    if len(selected):
        corr = xd[selected].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(selected)), selected, rotation=90, fontsize=7)
        ax.set_yticks(range(len(selected)), selected, fontsize=7)
        fig.colorbar(image, ax=ax)
        ax.set_title('Development feature correlations')
        saveplot(artifact_path(out, 'correlation_heatmap.png'))
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
            max_depth=12, n_jobs=-1, random_state=SEED),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=150,
            max_leaf_nodes=15, l2_regularization=1, random_state=SEED)}
    if args.lightgbm:
        from lightgbm import LGBMClassifier
        models['lightgbm'] = LGBMClassifier(n_estimators=200, learning_rate=.05,
            num_leaves=15, reg_lambda=1, random_state=SEED, n_jobs=4, verbosity=-1)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
    splits = list(cv.split(xd, yd))
    comparison, candidates = [], {}
    development_predictions = pd.DataFrame({
        'SK_ID_CURR': frame.iloc[dev].SK_ID_CURR.to_numpy(), 'TARGET': yd.to_numpy()})
    fold_ids = np.zeros(len(dev), dtype=int)
    for fold, (_, val) in enumerate(splits):
        fold_ids[val] = fold
    development_predictions['fold'] = fold_ids
    for name, estimator in models.items():
        print(f'Evaluating {name}...', flush=True)
        candidate = pipeline(xd, estimator)
        # Each fold learns its own imputations, scaling, and categories.
        oof = cross_val_predict(candidate, xd, yd, cv=splits,
                               method='predict_proba', n_jobs=1)[:, 1]
        fold_aucs = [roc_auc_score(yd.iloc[val], oof[val]) for _, val in splits]
        comparison.append({'model': name, 'cv_auc_mean': np.mean(fold_aucs),
            'cv_auc_std': np.std(fold_aucs), 'oof_auc': roc_auc_score(yd, oof),
            'fold_aucs': json.dumps(fold_aucs)})
        candidates[name] = candidate
        development_predictions[name] = oof
    table = pd.DataFrame(comparison).sort_values('cv_auc_mean', ascending=False)
    table.to_csv(artifact_path(out, 'model_comparison.csv'), index=False)
    best_name = table.iloc[0]['model']
    development_predictions['probability'] = development_predictions[best_name]
    development_predictions.to_csv(artifact_path(out, 'development_predictions.csv'), index=False)
    best = candidates[best_name].fit(xd, yd)
    probabilities = best.predict_proba(xh)[:, 1]
    predicted = (probabilities >= .5).astype(int)
    metrics = {'run_created_at': datetime.now(TZ).isoformat(), 'data_source': 'SYNTHETIC DEMO' if args.demo else str(args.train),
        'selected_model': best_name, 'selection_metric': 'development CV mean ROC-AUC',
        'seed': SEED, 'folds': args.folds, 'development_rows': len(dev),
        'holdout_rows': len(hold), 'threshold': .5,
        'holdout_auc': roc_auc_score(yh, probabilities),
        'holdout_average_precision': average_precision_score(yh, probabilities),
        'precision': precision_score(yh, predicted, zero_division=0),
        'recall': recall_score(yh, predicted, zero_division=0),
        'f1': f1_score(yh, predicted, zero_division=0)}
    (artifact_path(out, 'metrics.json')).write_text(json.dumps(metrics, indent=2)+'\n')
    pd.DataFrame({'SK_ID_CURR': frame.iloc[hold].SK_ID_CURR.to_numpy(),
        'TARGET': yh.to_numpy(), 'probability': probabilities,
        'predicted': predicted}).to_csv(artifact_path(out, 'holdout_predictions.csv'), index=False)
    RocCurveDisplay.from_predictions(yh, probabilities)
    saveplot(artifact_path(out, 'roc_curve.png'))
    PrecisionRecallDisplay.from_predictions(yh, probabilities)
    saveplot(artifact_path(out, 'precision_recall_curve.png'))
    ConfusionMatrixDisplay.from_predictions(yh, predicted, labels=[0, 1])
    saveplot(artifact_path(out, 'confusion_matrix.png'))
    # Interpretation uses development rows, avoiding repeated holdout decisions.
    sample = xd.sample(min(500, len(xd)), random_state=SEED)
    importance = permutation_importance(best, sample, yd.loc[sample.index],
        scoring='roc_auc', n_repeats=3, random_state=SEED, n_jobs=1)
    ranked = pd.Series(importance.importances_mean, index=x.columns).sort_values()
    ranked.rename('training_permutation_auc_drop').to_csv(artifact_path(out, 'feature_importance.csv'))
    ranked.tail(15).plot.barh(title='Development permutation importance (in-sample)')
    plt.xlabel('ROC-AUC decrease after shuffling')
    saveplot(artifact_path(out, 'feature_importance.png'))
    shap_status = 'Not requested; use --shap to explain the selected model.'
    if args.shap:
        import shap
        transformed = best['preprocess'].transform(sample.iloc[:100])
        names = best['preprocess'].get_feature_names_out()
        model = best['model']
        if best_name == 'logistic_regression':
            explainer = shap.LinearExplainer(model, transformed)
        else:
            explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        if np.asarray(values).ndim == 3:
            values = values[:, :, 1]
        shap.summary_plot(values, transformed, feature_names=names, show=False)
        saveplot(artifact_path(out, 'shap_summary.png'))
        shap_status = 'SHAP summary generated on development rows.'
    # Persist evaluation model separately from the all-data submission model.
    joblib.dump(best, artifact_path(out, 'evaluation_model.joblib'))
    submission_status = 'No application_test.csv supplied; submission not generated.'
    test, evaluation_test_probability = None, None
    if args.test is not None:
        if args.demo:
            raise ValueError('Demo data cannot be used to generate a Kaggle submission.')
        test = pd.read_csv(args.test)
        if 'TARGET' in test or 'SK_ID_CURR' not in test:
            raise ValueError('Test data needs SK_ID_CURR and must not include TARGET.')
        if test.SK_ID_CURR.isna().any() or test.SK_ID_CURR.duplicated().any():
            raise ValueError('Test applicant IDs must be nonmissing and unique.')
        if set(test.SK_ID_CURR) & set(frame.SK_ID_CURR):
            raise ValueError('Train and test applicant IDs overlap.')
        xt = features(test)
        if set(x.columns) != set(xt.columns):
            raise ValueError('Train/test feature columns differ.')
        evaluation_test_probability = best.predict_proba(xt[x.columns])[:, 1]
        final = clone(best).fit(x, y)
        pd.DataFrame({'SK_ID_CURR': test.SK_ID_CURR,
            'TARGET': final.predict_proba(xt[x.columns])[:, 1]}).to_csv(artifact_path(out, 'submission.csv'), index=False)
        joblib.dump(final, artifact_path(out, 'submission_model.joblib'))
        submission_status = 'submission.csv generated; no Kaggle upload performed.'
    monitoring_status = 'Monitoring diagnostics not requested; use --monitoring.'
    if args.monitoring:
        from src.monitoring import analyse
        monitoring = analyse(out, frame.iloc[dev], frame.iloc[hold],
            development_predictions.probability.to_numpy(), probabilities,
            test_frame=test, test_probability=evaluation_test_probability,
            reference_scores=probabilities)
        monitoring_status = (
            f'Development F2-selected cutoff: {monitoring["threshold"]:.6f}. '
            'Calibration, cutoff, data-quality, and optional train/test drift diagnostics saved. '
            'The holdout was previously inspected; these are follow-up diagnostics, not fresh final validation.')
    label = 'Synthetic smoke test' if args.demo else 'Home Credit application-table benchmark'
    (artifact_path(out, 'report.md')).write_text(f'# {label}\n\n'
        f'Selected model: {best_name}. Holdout ROC-AUC: {metrics["holdout_auc"]:.4f}.\n\n'
        'Models are selected on stratified development CV; only the winner is evaluated on a separate 20% holdout. '
        'Preprocessing is fitted within each fold. Threshold is fixed at 0.5, not optimized on the holdout.\n\n'
        f'{shap_status}\n\n{submission_status}\n\n{monitoring_status}\n\n'
        'Limitations: random holdout is not a temporal/external validation set; feature importance is in-sample; '
        'secondary tables and hyperparameter search are not implemented. '
        'The data dictionary still needs a feature-availability/leakage review. '
        'Synthetic metrics do not measure Home Credit performance.\n')
    print(json.dumps(metrics, indent=2))
    print(f'Artifacts: {out.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/application_train.csv'))
    parser.add_argument('--test', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--demo', action='store_true', help='Synthetic smoke test only')
    parser.add_argument('--lightgbm', action='store_true', help='Include LightGBM in comparison')
    parser.add_argument('--shap', action='store_true', help='Generate selected-model SHAP summary')
    parser.add_argument('--monitoring', action='store_true',
        help='Development F2 cutoff, calibration checks, data quality, and optional test drift')
    args = parser.parse_args()
    if args.demo and args.test:
        parser.error('--demo cannot be combined with --test')
    run(args)


if __name__ == '__main__':
    main()
