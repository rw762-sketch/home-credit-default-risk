"""Matched development feature/algorithm ablations, with fixed bounded models."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.artifacts import artifact_path, new_run, TZ
from src.bureau import aggregate_bureau, join_bureau, read_bureau
from src.compare_bureau import development_folds, probability_metrics, sha256
from src.installments import aggregate_installments, join_installments, read_installments
from src.train import SEED, features, pipeline

STAGES = ['application_only', 'application_bureau', 'application_bureau_installments']


def save_checkpoint(path, signature, predictions, fold_rows):
    """Atomically preserve numeric predictions and metadata; no pickle/models."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, __state__=np.array(json.dumps({'signature': signature, 'fold_rows': fold_rows})),
                 **predictions)
    temporary.replace(path)


def validate_fold_records(predictions, rows, assignments, complete=False):
    """Reject inconsistent predictions/metrics before reusing or skipping fits."""
    folds = set(assignments.fold.unique())
    for p in predictions.values():
        if p.shape != (len(assignments),):
            raise ValueError('Saved prediction row count differs.')
        finite = np.isfinite(p)
        if np.isinf(p).any() or ((p[finite] < 0) | (p[finite] > 1)).any():
            raise ValueError('Saved probabilities must be in [0, 1] or missing for unfinished folds.')
    completed = set()
    for row in rows:
        key = (row['variant'], row['fold'])
        if key in completed:
            raise ValueError('Saved fold metrics contain a duplicate variant/fold.')
        if row['variant'] not in predictions or row['fold'] not in folds:
            raise ValueError('Saved fold metrics contain an unexpected variant/fold.')
        completed.add(key)
        p = predictions[row['variant']]
        mask = assignments.fold.eq(row['fold']).to_numpy()
        if not mask.any() or not np.isfinite(p[mask]).all():
            raise ValueError('Saved completed fold has missing predictions.')
        if row['validation_rows'] != int(mask.sum()) or row['training_rows'] != int((~mask).sum()):
            raise ValueError('Saved fold row counts disagree with assignments.')
        if not np.isfinite(row['seconds']) or row['seconds'] < 0 or row['iterations'] < 1:
            raise ValueError('Saved fold runtime/iteration metadata is invalid.')
        for name, value in probability_metrics(assignments.TARGET[mask], p[mask]).items():
            if not np.isclose(value, row[name], rtol=0, atol=1e-10):
                raise ValueError('Saved predictions disagree with completed fold metrics.')
    for name, p in predictions.items():
        for fold in folds:
            if (name, fold) not in completed:
                mask = assignments.fold.eq(fold).to_numpy()
                if complete or not np.isnan(p[mask]).all():
                    raise ValueError('Saved predictions have an unrecorded or incomplete fold.')


def load_checkpoint(path, signature, assignments):
    with np.load(path, allow_pickle=False) as saved:
        state = json.loads(str(saved['__state__']))
        if state['signature'] != signature:
            raise ValueError('Checkpoint inputs, code, cohort, dependency versions, or model configuration changed.')
        predictions = {name: saved[name].copy() for name in saved.files if name != '__state__'}
    rows = state['fold_rows']
    validate_fold_records(predictions, rows, assignments)
    return predictions, rows


def operating_points(ids, labels, probability):
    """Exact top-k ranks; deterministic applicant-ID tie break, no label tuning."""
    frame = pd.DataFrame({'id': np.asarray(ids), 'target': np.asarray(labels), 'p': probability})
    ranked = frame.sort_values(['p', 'id'], ascending=[False, True])
    positives = int(frame.target.sum())
    rows = []
    policies = [(f'top_{fraction:.0%}', ranked.iloc[:max(1, round(len(frame)*fraction))])
                for fraction in [.05, .1, .2]]
    policies.append(('score_ge_0.5', frame.loc[frame.p.ge(.5)]))
    for name, flagged in policies:
        tp, count = int(flagged.target.sum()), len(flagged)
        rows.append({'policy': name, 'rows': len(frame), 'positives': positives,
                     'flagged': count, 'true_positive': tp, 'false_positive': count-tp,
                     'false_negative': positives-tp,
                     'precision': tp/count if count else 0., 'recall': tp/positives if positives else 0.})
    return rows


def cached_bureau_run(folder, assignments, train, archive, max_iter):
    """Reuse only identical inputs, cohort, labels, folds, code, and HGB settings."""
    folder = Path(folder)
    manifest = json.loads(artifact_path(folder, 'manifest.json').read_text())
    metrics = json.loads(artifact_path(folder, 'metrics.json').read_text())
    if metrics['model_parameters'] != {'max_iter': max_iter, 'max_leaf_nodes': 15,
                                       'l2_regularization': 1, 'random_state': SEED}:
        raise ValueError('Cached HGB configuration differs from this experiment.')
    for path in [train, archive]:
        if manifest['inputs'].get(str(path.resolve())) != sha256(path):
            raise ValueError(f'Cached input differs: {path}')
    for name in ['train.py', 'bureau.py', 'compare_bureau.py']:
        if manifest['code'].get(name) != sha256(Path(__file__).with_name(name)):
            raise ValueError(f'Cached feature/training code differs: {name}')
    saved = pd.read_csv(artifact_path(folder, 'fold_assignments.csv'))
    pd.testing.assert_frame_equal(saved, assignments, check_dtype=False)
    if sha256(artifact_path(folder, 'fold_assignments.csv')) != manifest['fold_assignments_sha256']:
        raise ValueError('Cached assignments failed their manifest hash.')
    predictions = pd.read_csv(artifact_path(folder, 'development_predictions.csv'))
    pd.testing.assert_frame_equal(predictions[['SK_ID_CURR', 'TARGET', 'fold']], assignments, check_dtype=False)
    mapping = {'application_only': 'hgb_application_only',
               'application_plus_bureau': 'hgb_application_bureau'}
    result = {}
    for old, new in mapping.items():
        p = predictions[old].to_numpy()
        if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
            raise ValueError('Cached probabilities must be finite and in [0, 1].')
        summary = next(s for s in metrics['summary'] if s['variant'] == old)
        for name, value in probability_metrics(assignments.TARGET, p).items():
            if not np.isclose(value, summary[f'oof_{name}'], rtol=0, atol=1e-10):
                raise ValueError('Cached predictions disagree with saved measured metrics.')
        result[new] = p
    folds = pd.read_csv(artifact_path(folder, 'fold_metrics.csv'))
    folds['variant'] = folds.variant.map(mapping)
    if folds.variant.isna().any() or len(folds) != assignments.fold.nunique()*2:
        raise ValueError('Cached fold metrics do not match the requested two variants.')
    folds['cached'] = True
    validate_fold_records(result, folds.to_dict('records'), assignments, complete=True)
    return result, folds.to_dict('records'), {
        'source': str(folder.resolve()), 'manifest_sha256': sha256(artifact_path(folder, 'manifest.json')),
        'predictions_sha256': sha256(artifact_path(folder, 'development_predictions.csv'))}


def run(args):
    started = perf_counter()
    if not 2 <= args.folds <= 5 or not 1 <= args.max_iter <= 150 or not 1 <= args.lgbm_estimators <= 200 or args.threads < 1:
        raise ValueError('Use 2-5 folds, 1-150 HGB iterations, 1-200 LightGBM trees, and positive threads.')
    resume = getattr(args, 'resume', False)
    if resume and args.output is None:
        raise ValueError('--resume requires the existing --output directory.')
    lightgbm_class = None
    if args.include_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            lightgbm_class = LGBMClassifier
        except (ImportError, OSError) as exc:
            raise RuntimeError('LightGBM or its native OpenMP runtime is unavailable. '
                               'Install the optional dependency/runtime or omit --include-lightgbm. '
                               'On macOS an existing trusted libomp directory may be supplied '
                               'using process-local DYLD_FALLBACK_LIBRARY_PATH.') from exc
    selected, fold_ids, full_rows, hold_rows = development_folds(
        pd.read_csv(args.train), args.folds, args.max_development_rows)
    assignments = selected[['SK_ID_CURR', 'TARGET']].copy()
    assignments['fold'] = fold_ids
    predictions, fold_rows, cache = {}, [], None
    if args.reuse_bureau_run:
        predictions, fold_rows, cache = cached_bureau_run(
            args.reuse_bureau_run, assignments, args.train, args.archive, args.max_iter)
        print(f'Verified and reused {len(predictions)} HGB OOF variants; no duplicate fits.', flush=True)
    bureau, bureau_audit = aggregate_bureau(read_bureau(args.archive), selected.SK_ID_CURR)
    with_bureau = join_bureau(selected, bureau)
    print('Reading scoped installment history in 250,000-row chunks...', flush=True)
    raw_installments, read_audit = read_installments(args.archive, selected.SK_ID_CURR)
    installments, installment_audit = aggregate_installments(raw_installments)
    del raw_installments
    installment_audit.update(read_audit)
    combined = join_installments(with_bureau, installments)
    installment_audit['selected_applicants_without_eligible_history'] = int(combined.INST_EVENT_COUNT.eq(0).sum())
    variants = {'application_only': features(selected), 'application_bureau': features(with_bureau),
                'application_bureau_installments': features(combined)}
    del selected, with_bureau, combined, bureau, installments
    out = args.output or new_run('history-comparison')
    if args.output and not resume:
        out.mkdir(parents=True, exist_ok=False)
    code_names = ['compare_history.py', 'installments.py', 'bureau.py', 'compare_bureau.py', 'train.py', 'artifacts.py']
    input_hashes = {str(p.resolve()): sha256(p) for p in [args.train, args.archive]}
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in code_names}
    versions = {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl']
                + (['lightgbm'] if args.include_lightgbm else [])}
    signature_content = {'inputs': input_hashes, 'code': code_hashes, 'versions': versions,
                         'assignments': assignments.to_dict('list'), 'folds': args.folds,
                         'max_iter': args.max_iter, 'lgbm_estimators': args.lgbm_estimators,
                         'include_lightgbm': args.include_lightgbm, 'threads': args.threads}
    signature = hashlib.sha256(json.dumps(signature_content, sort_keys=True).encode()).hexdigest()
    checkpoint = artifact_path(out, 'checkpoint.npz')
    if resume:
        saved_assignments = pd.read_csv(artifact_path(out, 'fold_assignments.csv'))
        pd.testing.assert_frame_equal(saved_assignments, assignments, check_dtype=False)
        predictions, fold_rows = load_checkpoint(checkpoint, signature, assignments)
        print(f'Resuming {len(fold_rows)} verified completed variant/folds.', flush=True)
    else:
        assignments.to_csv(artifact_path(out, 'fold_assignments.csv'), index=False)
        save_checkpoint(checkpoint, signature, predictions, fold_rows)
    algorithms = ['hgb', 'lgbm'] if args.include_lightgbm else ['hgb']
    labels = assignments.TARGET.to_numpy()
    hgb_params = {'max_iter': args.max_iter, 'max_leaf_nodes': 15, 'l2_regularization': 1, 'random_state': SEED}
    lgbm_params = {'n_estimators': args.lgbm_estimators, 'learning_rate': .05, 'num_leaves': 15,
                   'reg_lambda': 1, 'random_state': SEED, 'n_jobs': args.threads, 'verbosity': -1,
                   'deterministic': True, 'force_col_wise': True}
    for algorithm in algorithms:
        for stage in STAGES:
            name = f'{algorithm}_{stage}'
            if name in predictions and np.isfinite(predictions[name]).all():
                continue
            p = predictions.get(name, np.full(len(labels), np.nan))
            predictions[name] = p
            with threadpool_limits(limits=args.threads):
                for fold in sorted(np.unique(fold_ids)):
                    train, val = fold_ids != fold, fold_ids == fold
                    if any(row['variant'] == name and row['fold'] == int(fold) for row in fold_rows):
                        continue
                    tick = perf_counter()
                    estimator = (HistGradientBoostingClassifier(**hgb_params) if algorithm == 'hgb'
                                 else lightgbm_class(**lgbm_params))
                    model = pipeline(variants[stage], estimator).fit(variants[stage].loc[train], labels[train])
                    p[val] = model.predict_proba(variants[stage].loc[val])[:, 1]
                    row = {'variant': name, 'fold': int(fold), 'training_rows': int(train.sum()),
                           'validation_rows': int(val.sum()), 'cached': False,
                           'iterations': int(model['model'].n_iter_),
                           'seconds': round(perf_counter()-tick, 3), **probability_metrics(labels[val], p[val])}
                    fold_rows.append(row)
                    save_checkpoint(checkpoint, signature, predictions, fold_rows)
                    pd.DataFrame(fold_rows).to_csv(artifact_path(out, 'fold_metrics.csv'), index=False)
                    print(f'{name} fold {fold}: AUC={row["roc_auc"]:.6f}, AP={row["average_precision"]:.6f}, '
                          f'{row["seconds"]:.1f}s', flush=True)
                    del model
            predictions[name] = p
            pd.DataFrame(fold_rows).to_csv(artifact_path(out, 'fold_metrics.csv'), index=False)
            # Save completed variants so a connection interruption loses no results.
            partial = assignments.assign(**predictions)
            partial.to_csv(artifact_path(out, 'development_predictions.csv'), index=False)
    fold_table = pd.DataFrame(fold_rows)
    summary, points = [], []
    for name, p in predictions.items():
        fold_scores = fold_table.loc[fold_table.variant.eq(name)]
        stage = name.split('_', 1)[1]
        summary.append({'variant': name, 'features': variants[stage].shape[1],
                        'mean_fold_auc': float(fold_scores.roc_auc.mean()),
                        'fold_auc_std': float(fold_scores.roc_auc.std(ddof=0)),
                        'fit_seconds': float(fold_scores.seconds.sum()),
                        **{f'oof_{k}': v for k, v in probability_metrics(labels, p).items()}})
        for split, mask in [('pooled OOF', np.ones(len(labels), dtype=bool)),
                            *[(f'fold {fold}', fold_ids == fold) for fold in sorted(np.unique(fold_ids))]]:
            points.extend({'variant': name, 'split': split, **row} for row in
                          operating_points(assignments.SK_ID_CURR[mask], labels[mask], p[mask]))
    contrasts = [(f'{alg}_{STAGES[i]}', f'{alg}_{STAGES[i+1]}')
                 for alg in algorithms for i in [0, 1]]
    if args.include_lightgbm:
        contrasts += [(f'hgb_{stage}', f'lgbm_{stage}') for stage in STAGES]
    paired = []
    for base, candidate in contrasts:
        a = fold_table[fold_table.variant.eq(base)].set_index('fold')
        b = fold_table[fold_table.variant.eq(candidate)].set_index('fold')
        for fold in a.index:
            paired.append({'baseline': base, 'candidate': candidate, 'fold': int(fold),
                           **{f'{metric}_delta': float(b.loc[fold, metric]-a.loc[fold, metric])
                              for metric in ['roc_auc', 'average_precision', 'brier', 'log_loss']}})
    metadata = {'created_at': datetime.now(TZ).isoformat(), 'evaluation': 'Matched development CV; not fresh validation.',
                'selected_development_rows': len(labels), 'full_development_rows': full_rows,
                'excluded_holdout_rows': hold_rows, 'positive_rows': int(labels.sum()),
                'positive_prevalence': float(labels.mean()), 'folds': args.folds, 'seed': SEED,
                'threads': args.threads, 'max_development_rows': args.max_development_rows,
                'hgb_parameters': hgb_params, 'lgbm_parameters': lgbm_params if args.include_lightgbm else None,
                'summary': summary, 'reused_bureau_run': cache,
                'current_run_wall_seconds': round(perf_counter()-started, 3),
                'limitations': ['All estimates use previously inspected development data; holdout is not scored.',
                                'Fixed configured models: HGB internal early stopping and LightGBM fixed tree budget differ.',
                                'Installment features describe observed past-payment tranches, not every unpaid obligation.',
                                'Ambiguous schedules/duplicate events are conservatively excluded; snapshots need provenance review.',
                                'Top-k policies are descriptive capacity comparisons, not tuned lending decisions.']}
    manifest = {'inputs': input_hashes, 'code': code_hashes, 'versions': versions,
                'checkpoint_signature': signature,
                'command': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'native_library_fallback': os.environ.get('DYLD_FALLBACK_LIBRARY_PATH'),
                'fold_assignments_sha256': sha256(artifact_path(out, 'fold_assignments.csv')),
                'predictions_sha256': sha256(artifact_path(out, 'development_predictions.csv'))}
    for name, data in [('metrics.json', metadata), ('manifest.json', manifest),
                       ('bureau_audit.json', bureau_audit), ('installments_audit.json', installment_audit),
                       ('feature_schema.json', {name: x.columns.tolist() for name, x in variants.items()})]:
        artifact_path(out, name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    for name, data in [('comparison.csv', pd.DataFrame(summary)), ('operating_points.csv', pd.DataFrame(points)),
                       ('paired_deltas.csv', pd.DataFrame(paired)), ('fold_metrics.csv', fold_table)]:
        data.to_csv(artifact_path(out, name), index=False)
    lines = ['# Staged credit-history and model comparison', '',
             f'{len(labels):,} development applications; {int(labels.sum()):,} positives ({labels.mean():.2%}); '
             f'{args.folds} matched folds, seed {SEED}, {args.threads} native threads. Holdout excluded.', '',
             '| Variant | Mean fold AUC | OOF AP | OOF Brier | Fit seconds (including reused fits) |',
             '| --- | ---: | ---: | ---: | ---: |',
             *[f'| {s["variant"]} | {s["mean_fold_auc"]:.6f} | {s["oof_average_precision"]:.6f} | '
               f'{s["oof_brier"]:.6f} | {s["fit_seconds"]:.1f} |' for s in summary], '',
             'Top-risk policies (pooled OOF; applicant ID breaks score ties):', '',
             '| Variant | Policy | TP | FP | Precision | Recall |', '| --- | --- | ---: | ---: | ---: | ---: |',
             *[f'| {p["variant"]} | {p["policy"]} | {p["true_positive"]} | {p["false_positive"]} | '
               f'{p["precision"]:.2%} | {p["recall"]:.2%} |' for p in points
               if p['split'] == 'pooled OOF' and p['policy'] != 'score_ge_0.5'], '',
             *[f'- {x}' for x in metadata['limitations']], '',
             'Per-fold metrics, paired feature/algorithm deltas, operating-point counts, exact assignments, '
             'predictions, audits, feature schema, code/input hashes and versions are saved alongside this report.', '']
    artifact_path(out, 'report.md').write_text('\n'.join(lines))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f'Artifacts: {out.resolve()}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/application_train.csv'))
    parser.add_argument('--archive', type=Path, required=True, help='Local Home Credit ZIP with bureau/installment tables.')
    parser.add_argument('--reuse-bureau-run', type=Path, help='Verified same-cohort bureau comparison; saves duplicate HGB fits.')
    parser.add_argument('--include-lightgbm', action='store_true')
    parser.add_argument('--max-development-rows', type=int, default=30000, help='0 explicitly uses all development rows.')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--lgbm-estimators', type=int, default=200)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume verified fold checkpoints in --output; reject changed inputs/code/configuration.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
