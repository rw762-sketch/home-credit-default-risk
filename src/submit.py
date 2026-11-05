"""Freeze the full-feature pipeline, fit it on every labeled row, and score the test table.

This is the one-shot independent evaluation: the configuration below was fixed
from development evidence before any test score is seen. It writes a Kaggle
submission file locally; uploading it is a separate, manual step.
"""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.application import add_application_features
from src.artifacts import TZ, artifact_path, new_run
from src.bureau import aggregate_bureau, join_bureau, read_bureau
from src.compare_bureau import sha256
from src.compare_history import INNER_VALIDATION_FRACTION, fit_fold
from src.installments import aggregate_installments, join_installments, read_installments
from src.previous import aggregate_previous, join_previous, read_previous
from src.train import SEED, features, validate

STAGE = 'application_bureau_installments_ext_previous'


def build_frame(applications, archive):
    """Add every history family used by the full-feature development stage.

    Each aggregate is computed per applicant from that applicant's own records,
    so scoring train and test rows together shares no statistics between them.
    """
    ids = applications.SK_ID_CURR
    audits = {}
    bureau, audits['bureau'] = aggregate_bureau(read_bureau(archive), ids)
    frame = join_bureau(applications, bureau)
    raw, read_audit = read_installments(archive, ids)
    installments, audits['installments'] = aggregate_installments(raw)
    audits['installments'].update(read_audit)
    frame = join_installments(frame, installments)
    frame = add_application_features(frame)
    raw, read_audit = read_previous(archive, ids)
    previous, audits['previous'] = aggregate_previous(raw)
    audits['previous'].update(read_audit)
    frame = join_previous(frame, previous)
    return frame, audits


def model_parameters(algorithm, learning_rate, max_rounds, threads):
    """The November 5 development protocol, frozen."""
    if algorithm == 'hgb':
        return {'max_iter': max_rounds, 'max_leaf_nodes': 15, 'l2_regularization': 1,
                'random_state': SEED, 'learning_rate': learning_rate, 'early_stopping': True,
                'validation_fraction': INNER_VALIDATION_FRACTION}, None
    return None, {'n_estimators': max_rounds, 'learning_rate': learning_rate, 'num_leaves': 15,
                  'reg_lambda': 1, 'random_state': SEED, 'n_jobs': threads, 'verbosity': -1,
                  'deterministic': True, 'force_col_wise': True}


def prepare(train, test, archive):
    """Validate the two tables and return aligned feature matrices."""
    validate(train, 5)
    if 'TARGET' in test or 'SK_ID_CURR' not in test:
        raise ValueError('Test data needs SK_ID_CURR and must not include TARGET.')
    if test.SK_ID_CURR.isna().any() or test.SK_ID_CURR.duplicated().any():
        raise ValueError('Test applicant IDs must be nonmissing and unique.')
    if set(train.SK_ID_CURR) & set(test.SK_ID_CURR):
        raise ValueError('Train and test applicant IDs overlap.')
    combined = pd.concat([train, test], ignore_index=True)
    labeled = combined.TARGET.notna().to_numpy()
    frame, audits = build_frame(combined, archive)
    x_train = features(frame.loc[labeled]).reset_index(drop=True)
    x_test = features(frame.loc[~labeled]).reset_index(drop=True)
    if set(x_train.columns) != set(x_test.columns):
        raise ValueError('Train/test feature columns differ.')
    x_test = x_test[x_train.columns]
    y = frame.loc[labeled, 'TARGET'].astype(int).to_numpy()
    test_ids = frame.loc[~labeled, 'SK_ID_CURR'].to_numpy()
    return x_train, y, x_test, test_ids, audits


def run(args):
    started = perf_counter()
    if args.algorithm not in {'lgbm', 'hgb'} or args.max_rounds < 1 or args.patience < 1 or args.threads < 1:
        raise ValueError('Use lgbm/hgb, positive rounds, patience and threads.')
    if not 0 < args.learning_rate <= 1:
        raise ValueError('--learning-rate must be in (0, 1].')
    lightgbm_class = None
    if args.algorithm == 'lgbm':
        from lightgbm import LGBMClassifier
        lightgbm_class = LGBMClassifier
    print('Building full-feature tables for labeled and test applicants...', flush=True)
    x_train, y, x_test, test_ids, audits = prepare(
        pd.read_csv(args.train), pd.read_csv(args.test), args.archive)
    hgb_params, lgbm_params = model_parameters(args.algorithm, args.learning_rate,
                                               args.max_rounds, args.threads)
    print(f'Fitting {args.algorithm} on {len(y):,} labeled rows ({x_train.shape[1]} features)...', flush=True)
    tick = perf_counter()
    model, iterations = fit_fold(x_train, y, args.algorithm, hgb_params, lgbm_params,
                                 args.patience, lightgbm_class, SEED)
    fit_seconds = perf_counter() - tick
    probability = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Test probabilities must be finite and in [0, 1].')
    out = args.output or new_run('submission')
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': probability})
    submission.to_csv(artifact_path(out, 'submission.csv'), index=False)
    joblib.dump(model, artifact_path(out, 'submission_model.joblib'))
    reasons_status = 'Reason codes not requested.'
    if args.reasons:
        from src.reasons import reason_codes, reason_summary
        print('Computing per-applicant reason codes...', flush=True)
        codes = reason_codes(model, x_test, test_ids, top_k=args.reasons)
        if args.reason_cutoff is not None:
            codes = codes.loc[codes.probability >= args.reason_cutoff]
        codes.to_csv(artifact_path(out, 'reason_codes.csv'), index=False)
        reason_summary(codes, args.reasons).to_csv(artifact_path(out, 'reason_summary.csv'), index=False)
        scope = ('all test applicants' if args.reason_cutoff is None
                 else f'{len(codes):,} test applicants at or above PD {args.reason_cutoff}')
        reasons_status = f'Top-{args.reasons} reason codes saved for {scope}.'
    quantiles = {f'p{int(q*100):02d}': float(np.quantile(probability, q)) for q in [.05, .25, .5, .75, .95]}
    configuration = {'stage': STAGE, 'algorithm': args.algorithm, 'learning_rate': args.learning_rate,
                     'max_rounds': args.max_rounds, 'patience': args.patience, 'seed': SEED,
                     'hgb_parameters': hgb_params, 'lgbm_parameters': lgbm_params}
    configuration_sha256 = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    metrics = {'created_at': datetime.now(TZ).isoformat(),
               'evaluation': 'Frozen pipeline scored once on the unlabeled competition test table; '
                             'its AUC comes only from the Kaggle leaderboard.',
               'configuration': configuration, 'configuration_sha256': configuration_sha256,
               'labeled_rows': int(len(y)), 'labeled_prevalence': float(y.mean()),
               'test_rows': int(len(test_ids)), 'features': int(x_train.shape[1]),
               'search_iterations': int(model.audit_search_iterations_),
               'selected_iterations': int(model.audit_selected_iterations_),
               'refit_iterations': int(iterations), 'fit_seconds': round(fit_seconds, 1),
               'test_mean_probability': float(probability.mean()), 'test_probability_quantiles': quantiles,
               'kaggle_score': None, 'wall_seconds': round(perf_counter()-started, 1)}
    code_names = ['submit.py', 'compare_history.py', 'train.py', 'bureau.py', 'installments.py',
                  'previous.py', 'application.py', 'reasons.py']
    manifest = {'inputs': {str(p.resolve()): sha256(p) for p in [args.train, args.test, args.archive]},
                'code': {n: sha256(Path(__file__).with_name(n)) for n in code_names},
                'versions': {n: version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'joblib']
                             + (['lightgbm'] if args.algorithm == 'lgbm' else [])},
                'command': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'submission_sha256': sha256(artifact_path(out, 'submission.csv'))}
    for name, data in [('metrics.json', metrics), ('manifest.json', manifest),
                       *[(f'{k}_audit.json', v) for k, v in audits.items()]]:
        artifact_path(out, name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    submission_file = artifact_path(out, 'submission.csv')
    report = [
        '# Frozen full-feature submission', '',
        f'{args.algorithm} on `{STAGE}`: {x_train.shape[1]} features, {len(y):,} labeled rows, '
        f'{len(test_ids):,} test rows. Configuration hash `{configuration_sha256[:12]}`.', '',
        f'Inner early stopping searched {model.audit_search_iterations_} rounds and selected '
        f'{model.audit_selected_iterations_}; the final model was refitted on all labeled rows. '
        f'Fit took {fit_seconds:.0f} seconds.', '',
        f'Mean predicted test probability: {probability.mean():.4f} (labeled default rate {y.mean():.4f}). '
        'A large gap would suggest drift or a pipeline error; it is not a performance measure.', '',
        reasons_status, '',
        'This file has **not** been uploaded. Submit it once, then record the public and private '
        'leaderboard AUC in metrics.json (`kaggle_score`) and the README:', '',
        '```bash',
        f'kaggle competitions submit -c home-credit-default-risk -f "{submission_file}" '
        '-m "Frozen full-feature pipeline"',
        '```', '',
        'Submitting more than once and keeping the best score would turn the leaderboard into another '
        'tuning set. The configuration hash above identifies the pipeline that was frozen.', '']
    artifact_path(out, 'report.md').write_text('\n'.join(report))
    print('\n'.join(report[2:6]), flush=True)
    print(f'Artifacts: {out.resolve()}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/application_train.csv'))
    parser.add_argument('--test', type=Path, default=Path('data/raw/application_test.csv'))
    parser.add_argument('--archive', type=Path, required=True,
                        help='Local Home Credit ZIP with bureau, installment and previous-application tables.')
    parser.add_argument('--algorithm', choices=['lgbm', 'hgb'], default='lgbm',
                        help='LightGBM tied HGB on development AUC and fits about 4x faster.')
    parser.add_argument('--learning-rate', type=float, default=.05)
    parser.add_argument('--max-rounds', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--reasons', type=int, default=3,
                        help='Top-k reason codes per test applicant (LightGBM only); 0 skips them.')
    parser.add_argument('--reason-cutoff', type=float,
                        help='Only save reason codes for applicants at or above this PD (e.g. the cost-based cutoff).')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.reasons and args.algorithm != 'lgbm':
        parser.error('--reasons needs --algorithm lgbm (it uses LightGBM tree contributions).')
    run(args)


if __name__ == '__main__':
    main()
