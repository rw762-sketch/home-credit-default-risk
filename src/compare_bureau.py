"""Bounded, matched-fold bureau feature ablation; development evidence only."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.artifacts import artifact_path, new_run, TZ
from src.bureau import FEATURE_COLUMNS, aggregate_bureau, join_bureau, read_bureau
from src.train import SEED, features, pipeline, validate


def development_folds(frame, folds=5, max_rows=30000):
    """Recreate the original split/folds, then subsample within label/fold strata.

    The full 20% holdout stays excluded. Input row order must match the original
    CSV; source hashes and selected applicant/fold assignments are saved per run.
    """
    validate(frame, folds)
    if max_rows < 0:
        raise ValueError('max_rows must be nonnegative; 0 means all development rows.')
    dev, hold = train_test_split(np.arange(len(frame)), test_size=.2,
                                stratify=frame.TARGET, random_state=SEED)
    labels = frame.TARGET.iloc[dev].to_numpy()
    if pd.Series(labels).value_counts().min() < folds:
        raise ValueError('Development split has too few minority samples for folds.')
    fold_ids = np.empty(len(dev), dtype=int)
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    for fold, (_, val) in enumerate(cv.split(dev, labels)):
        fold_ids[val] = fold
    keep = np.arange(len(dev))
    if 0 < max_rows < len(dev):
        strata = labels * folds + fold_ids
        keep, _ = train_test_split(keep, train_size=max_rows, stratify=strata,
                                   random_state=SEED)
        keep.sort()  # Preserve original development order for both variants.
    selected = frame.iloc[dev[keep]].copy().reset_index(drop=True)
    selected_folds = fold_ids[keep]
    for fold in range(folds):
        for mask in [selected_folds == fold, selected_folds != fold]:
            if selected.loc[mask, 'TARGET'].nunique() != 2:
                raise ValueError('Each training/validation fold needs both classes; increase row cap.')
    return selected, selected_folds, len(dev), len(hold)


def probability_metrics(y, probability):
    return {
        'roc_auc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'brier': float(brier_score_loss(y, probability)),
        'log_loss': float(log_loss(y, probability, labels=[0, 1])),
    }


def matched_predictions(variants, y, fold_ids, max_iter=150, threads=2):
    """Fit both variants from scratch on identical rows with the original model.

    Preprocessing is fold-fitted by the existing application pipeline. No search,
    holdout evaluation, threshold optimization, refit, or submission is performed.
    """
    labels = np.asarray(y)
    fold_ids = np.asarray(fold_ids)
    if len(labels) != len(fold_ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels and fold assignments must match and be binary.')
    if any(len(x) != len(labels) for x in variants.values()):
        raise ValueError('Every variant must contain the same application rows.')
    reference_index = next(iter(variants.values())).index
    if any(not x.index.equals(reference_index) for x in variants.values()):
        raise ValueError('Variant row indices must match in the same order.')
    predictions = {name: np.full(len(labels), np.nan) for name in variants}
    rows = []
    with threadpool_limits(limits=threads):
        for fold in sorted(np.unique(fold_ids)):
            training, validation = fold_ids != fold, fold_ids == fold
            if len(np.unique(labels[training])) != 2 or len(np.unique(labels[validation])) != 2:
                raise ValueError('Each training/validation fold needs both classes.')
            for name, x in variants.items():
                started = perf_counter()
                estimator = HistGradientBoostingClassifier(
                    max_iter=max_iter, max_leaf_nodes=15,
                    l2_regularization=1, random_state=SEED)
                model = pipeline(x, estimator).fit(x.loc[training], labels[training])
                probability = model.predict_proba(x.loc[validation])[:, 1]
                predictions[name][validation] = probability
                row = {'variant': name, 'fold': int(fold),
                       'training_rows': int(training.sum()),
                       'validation_rows': int(validation.sum()),
                       'iterations': int(model['model'].n_iter_),
                       'seconds': round(perf_counter() - started, 3),
                       **probability_metrics(labels[validation], probability)}
                rows.append(row)
                print(f'{name} fold {fold}: AUC={row["roc_auc"]:.6f}, '
                      f'AP={row["average_precision"]:.6f}, {row["seconds"]:.1f}s', flush=True)
    return predictions, pd.DataFrame(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    if not 2 <= args.folds <= 5 or not 1 <= args.max_iter <= 150 or args.threads < 1:
        raise ValueError('Use 2-5 folds, 1-150 iterations, and at least one thread.')
    frame = pd.read_csv(args.train)
    selected, fold_ids, development_rows, holdout_rows = development_folds(
        frame, args.folds, args.max_development_rows)
    del frame
    bureau = read_bureau(args.bureau)
    aggregates, audit = aggregate_bureau(bureau, selected.SK_ID_CURR)
    del bureau
    augmented = join_bureau(selected, aggregates)
    audit['selected_applicants_without_eligible_history'] = int(
        augmented.BUREAU_RECORD_COUNT.eq(0).sum())
    variants = {'application_only': features(selected),
                'application_plus_bureau': features(augmented)}
    out = args.output or new_run('bureau-comparison')
    if args.output:
        out.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
    assignments = selected[['SK_ID_CURR', 'TARGET']].copy()
    assignments['fold'] = fold_ids
    assignments.to_csv(artifact_path(out, 'fold_assignments.csv'), index=False)
    predictions, fold_metrics = matched_predictions(
        variants, selected.TARGET, fold_ids, args.max_iter, args.threads)
    fold_metrics.to_csv(artifact_path(out, 'fold_metrics.csv'), index=False)
    summary = []
    for name, probability in predictions.items():
        assignments[name] = probability
        scores = fold_metrics[fold_metrics.variant == name]
        summary.append({'variant': name, 'features': variants[name].shape[1],
                        'mean_fold_auc': float(scores.roc_auc.mean()),
                        'fold_auc_std': float(scores.roc_auc.std(ddof=0)),
                        'mean_fold_ap': float(scores.average_precision.mean()),
                        **{f'oof_{k}': v for k, v in
                           probability_metrics(selected.TARGET, probability).items()}})
    table = pd.DataFrame(summary)
    delta = fold_metrics.pivot(index='fold', columns='variant', values=['roc_auc', 'average_precision'])
    paired = [{'fold': int(fold),
               'auc_delta': float(delta.loc[fold, ('roc_auc', 'application_plus_bureau')] -
                                  delta.loc[fold, ('roc_auc', 'application_only')]),
               'ap_delta': float(delta.loc[fold, ('average_precision', 'application_plus_bureau')] -
                                 delta.loc[fold, ('average_precision', 'application_only')])}
              for fold in delta.index]
    metrics = {'created_at': datetime.now(TZ).isoformat(), 'data_source': str(args.train),
               'evaluation': 'Matched development CV ablation; not fresh final validation.',
               'seed': SEED, 'folds': args.folds, 'selected_development_rows': len(selected),
               'full_development_rows': development_rows, 'excluded_holdout_rows': holdout_rows,
               'max_development_rows': args.max_development_rows,
               'threads': args.threads, 'model': 'HistGradientBoostingClassifier',
               'model_parameters': {'max_iter': args.max_iter, 'max_leaf_nodes': 15,
                                    'l2_regularization': 1, 'random_state': SEED},
               'summary': summary, 'paired_fold_deltas': paired,
               'limitations': ['Development data and original holdout were previously inspected.',
                              'Row cap can change model early stopping and is not the original full-data benchmark.',
                              'Snapshot dates screen availability but cannot establish historical provenance.',
                              'No temporal/external evaluation, threshold optimization, or profitability claim.']}
    manifest = {
        'inputs': {str(p.resolve()): sha256(p) for p in [args.train, args.bureau]},
        'code': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('bureau.py'),
                                          Path(__file__).with_name('train.py'),
                                          Path(__file__).with_name('artifacts.py')]},
        'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl']},
        'command': {'train': str(args.train), 'bureau': str(args.bureau), 'folds': args.folds,
                    'max_development_rows': args.max_development_rows,
                    'max_iter': args.max_iter, 'threads': args.threads},
        'fold_assignments_sha256': sha256(artifact_path(out, 'fold_assignments.csv')),
    }
    assignments.to_csv(artifact_path(out, 'development_predictions.csv'), index=False)
    table.to_csv(artifact_path(out, 'comparison.csv'), index=False)
    for name, value in [('metrics.json', metrics), ('manifest.json', manifest),
                        ('bureau_audit.json', audit),
                        ('feature_schema.json', {name: x.columns.tolist() for name, x in variants.items()})]:
        artifact_path(out, name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    report = [
        '# Bureau history: matched development comparison', '',
        'Development evidence only. The original 20% holdout is excluded and is not scored.',
        f'{len(selected):,} applications; {args.folds} shared folds; seed {SEED}; '
        f'at most {args.max_iter} boosting iterations; {args.threads} native threads.', '',
        '| Variant | Mean fold ROC-AUC | OOF average precision | OOF Brier |',
        '| --- | ---: | ---: | ---: |',
        *[f'| {s["variant"]} | {s["mean_fold_auc"]:.6f} | '
          f'{s["oof_average_precision"]:.6f} | {s["oof_brier"]:.6f} |' for s in summary], '',
        f'Mean paired fold AUC change: {np.mean([p["auc_delta"] for p in paired]):+.6f}. '
        'Fold standard deviations are descriptive, not confidence intervals.', '',
        f'Bureau audit: {audit["scoped_rows"]:,} scoped records, '
        f'{audit["excluded_rows"]:,} excluded at the time boundary; '
        f'{audit["selected_applicants_without_eligible_history"]:,} applications have no eligible history.', '',
        'Only whitelisted bureau fields are used. Credit/update dates must be known and <= day 0; '
        'supplied actual closures after day 0 are excluded. Scheduled future end dates are not used. '
        'Signed debt balances are retained; zero-denominator ratios and unknown amounts stay missing. '
        'Counts and amount-coverage features distinguish absent history from unknown balances.', '',
        *[f'- {item}' for item in metrics['limitations']], '',
        'See the numbered manifest, fold assignments, per-fold metrics, paired deltas in metrics.json, '
        'feature schema, bureau audit, and OOF predictions alongside this report.', '',
    ]
    artifact_path(out, 'report.md').write_text('\n'.join(report))
    print(table.to_string(index=False), flush=True)
    print(f'Artifacts: {out.resolve()}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/application_train.csv'))
    parser.add_argument('--bureau', type=Path, required=True,
                        help='Explicit opt-in: bureau CSV or local dataset ZIP; no extraction/download.')
    parser.add_argument('--output', type=Path, help='New directory only; existing runs are never overwritten.')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max-development-rows', type=int, default=30000,
                        help='Stratified cap after original split/fold assignment; 0 uses all development rows.')
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
