"""Audit saved history-comparison OOF scores without fitting or scoring a holdout."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.compare_bureau import probability_metrics, sha256
from src.compare_history import validate_fold_records

KEYS = ['SK_ID_CURR', 'TARGET', 'fold']
BOUNDARY = (
    'Previously inspected development OOF predictions only. Bootstrap percentile '
    'intervals condition on the saved fitted scores and preserve each fold/class size. '
    'They omit model refitting, overlapping training-fold dependence, model selection '
    'and population shift; they are not generalization confidence intervals or '
    'significance tests. AUC/AP point estimates here are pooled OOF, distinct from '
    'the benchmark selection metric of mean fold AUC. '
    'Calibration is descriptive; no calibrator or cutoff is fitted. '
    'No holdout, test data or raw applicant features are read.'
)


def validate_predictions(frame):
    if not set(KEYS).issubset(frame.columns):
        raise ValueError('OOF data requires applicant ID, TARGET and fold columns.')
    if frame.empty or frame[KEYS].isna().any().any() or frame.SK_ID_CURR.duplicated().any():
        raise ValueError('Applicant IDs must be unique and cohort keys nonmissing.')
    if set(frame.TARGET.unique()) != {0, 1}:
        raise ValueError('TARGET must contain exactly both binary classes.')
    folds = frame.fold.to_numpy(dtype=float)
    if not np.isfinite(folds).all() or (folds < 0).any() or (folds != np.floor(folds)).any():
        raise ValueError('Fold identifiers must be nonnegative integers.')
    if frame.fold.nunique() < 2 or frame.groupby('fold').TARGET.nunique().ne(2).any():
        raise ValueError('At least two folds, each with both classes, are required.')
    variants = [c for c in frame if c not in KEYS]
    if not variants:
        raise ValueError('No OOF prediction variants found.')
    for name in variants:
        p = frame[name].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f'{name}: probabilities must be finite and in [0, 1].')
    return variants


def load_source(source):
    """Verify cohort hashes and independently recompute saved metrics before analysis."""
    paths = {name: artifact_path(source, name) for name in [
        'manifest.json', 'metrics.json', 'fold_assignments.csv',
        'development_predictions.csv', 'fold_metrics.csv']}
    manifest = json.loads(paths['manifest.json'].read_text())
    for filename, key in [('fold_assignments.csv', 'fold_assignments_sha256'),
                          ('development_predictions.csv', 'predictions_sha256')]:
        if sha256(paths[filename]) != manifest[key]:
            raise ValueError(f'Source hash mismatch: {filename}')
    frame = pd.read_csv(paths['development_predictions.csv'])
    variants = validate_predictions(frame)
    assignments = pd.read_csv(paths['fold_assignments.csv'])
    pd.testing.assert_frame_equal(frame[KEYS], assignments, check_dtype=False)
    metrics = json.loads(paths['metrics.json'].read_text())
    summaries = metrics['summary']
    if len(summaries) != len(variants) or {r['variant'] for r in summaries} != set(variants):
        raise ValueError('Source summary variant coverage differs from predictions.')
    if (metrics['selected_development_rows'] != len(frame)
            or metrics['positive_rows'] != int(frame.TARGET.sum())
            or metrics['folds'] != frame.fold.nunique()):
        raise ValueError('Source cohort counts disagree with predictions.')
    fold_rows = pd.read_csv(paths['fold_metrics.csv']).to_dict('records')
    validate_fold_records({v: frame[v].to_numpy() for v in variants},
                          fold_rows, assignments, complete=True)
    for row in summaries:
        for metric, value in probability_metrics(frame.TARGET, frame[row['variant']]).items():
            if not np.isclose(value, row[f'oof_{metric}'], rtol=0, atol=1e-10):
                raise ValueError('Source pooled metrics disagree with predictions.')
    return frame, paths


def calibration_rows(frame, variants, bins=10):
    """Fixed equal-width probability bins; include empty bins and exact 0/1 scores."""
    rows = []
    for variant in variants:
        for split, cohort in [('pooled OOF', frame), *[
                (f'fold {fold}', group) for fold, group in frame.groupby('fold', sort=True)]]:
            p = cohort[variant].to_numpy()
            labels = cohort.TARGET.to_numpy()
            groups = np.minimum((p*bins).astype(int), bins-1)
            for i in range(bins):
                mask = groups == i
                count = int(mask.sum())
                mean = float(p[mask].mean()) if count else None
                observed = float(labels[mask].mean()) if count else None
                rows.append({'variant': variant, 'split': split, 'bin': i,
                             'lower': i/bins, 'upper': (i+1)/bins, 'rows': count,
                             'positives': int(labels[mask].sum()), 'mean_probability': mean,
                             'observed_rate': observed,
                             'observed_minus_predicted': observed-mean if count else None})
    return rows


def paired_bootstrap(frame, baseline, candidate, repetitions=1000, seed=42):
    """Use the same applicant resample for both models within fold/label strata."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    variants = validate_predictions(frame)
    if baseline not in variants or candidate not in variants:
        raise ValueError('Both requested variants must exist in the saved OOF predictions.')
    if not 100 <= repetitions <= 10000:
        raise ValueError('Use 100-10000 bootstrap repetitions.')
    # Canonical order makes results invariant to CSV row order.
    frame = frame.sort_values('SK_ID_CURR').reset_index(drop=True)
    y = frame.TARGET.to_numpy()
    a, b = frame[baseline].to_numpy(), frame[candidate].to_numpy()
    strata = list(frame.groupby(['fold', 'TARGET'], sort=True).indices.values())
    rng = np.random.default_rng(seed)
    scorers = {'roc_auc': roc_auc_score, 'average_precision': average_precision_score}
    observed = {name: (float(score(y, a)), float(score(y, b))) for name, score in scorers.items()}
    deltas = {name: np.empty(repetitions) for name in scorers}
    # Sort fixed scores once. Integer applicant multiplicities are exactly the
    # ordinary resampled dataset, including tied scores, without repeated sorts.
    plans = [_ranking_plan(y, p) for p in [a, b]]
    for repeat in range(repetitions):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        weights = np.bincount(sample, minlength=len(y))
        scores = [_weighted_ranking_metrics(plan, weights) for plan in plans]
        for index, name in enumerate(scorers):
            deltas[name][repeat] = scores[1][index]-scores[0][index]
    return [{'metric': name, 'baseline': baseline, 'candidate': candidate,
             'baseline_score': observed[name][0], 'candidate_score': observed[name][1],
             'candidate_minus_baseline': observed[name][1]-observed[name][0],
             'conditional_percentile_2_5': float(np.quantile(values, .025)),
             'conditional_percentile_97_5': float(np.quantile(values, .975)),
             'repetitions': repetitions, 'seed': seed} for name, values in deltas.items()]


def _ranking_plan(labels, probability):
    order = np.argsort(-np.asarray(probability), kind='stable')
    sorted_scores = np.asarray(probability)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores))+1]
    return order, np.asarray(labels)[order], starts


def _weighted_ranking_metrics(plan, weights):
    """ROC trapezoids and noninterpolated AP after grouping equal score values."""
    order, labels, starts = plan
    ordered_weights = weights[order]
    positive = np.add.reduceat(ordered_weights*labels, starts)
    negative = np.add.reduceat(ordered_weights*(1-labels), starts)
    tp, fp = np.cumsum(positive), np.cumsum(negative)
    auc = np.sum(negative*(tp-.5*positive))/(tp[-1]*fp[-1])
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp) > 0)
    ap = np.sum(positive*precision)/tp[-1]
    return float(auc), float(ap)


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    frame, paths = load_source(args.source)
    paired = paired_bootstrap(frame, args.baseline, args.candidate, args.repetitions, args.seed)
    calibration = calibration_rows(frame, list(dict.fromkeys([args.baseline, args.candidate])))
    summary = {name: probability_metrics(frame.TARGET, frame[name])
               for name in dict.fromkeys([args.baseline, args.candidate])}
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(paired).to_csv(artifact_path(args.output, 'bootstrap_deltas.csv'), index=False)
    pd.DataFrame(calibration).to_csv(artifact_path(args.output, 'calibration.csv'), index=False)
    artifact_path(args.output, 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    manifest = {'boundary': BOUNDARY, 'rows': len(frame), 'positives': int(frame.TARGET.sum()),
                'source_files': {name: sha256(path) for name, path in paths.items()},
                'source_run': args.source.name,
                'audit_code_sha256': sha256(Path(__file__)),
                'metric_code_sha256': sha256(Path(__file__).with_name('compare_bureau.py')),
                'validation_code_sha256': sha256(Path(__file__).with_name('compare_history.py')),
                'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn']},
                'baseline': args.baseline, 'candidate': args.candidate,
                'repetitions': args.repetitions, 'seed': args.seed,
                'binning': '10 fixed equal-width bins; last bin includes probability 1'}
    artifact_path(args.output, 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# Saved development prediction audit', '', BOUNDARY, '',
             f'Rows: {len(frame):,}; positives: {int(frame.TARGET.sum()):,}.', '',
             f'Baseline: `{args.baseline}`. Candidate: `{args.candidate}`.', '',
             '| Metric | Baseline | Candidate | Change | Conditional 95% percentile interval |',
             '| --- | ---: | ---: | ---: | --- |']
    for row in paired:
        lines.append(f'| {row["metric"]} | {row["baseline_score"]:.6f} | '
                     f'{row["candidate_score"]:.6f} | {row["candidate_minus_baseline"]:+.6f} | '
                     f'[{row["conditional_percentile_2_5"]:+.6f}, '
                     f'{row["conditional_percentile_97_5"]:+.6f}] |')
    lines += ['', f'{args.repetitions} paired replicates; seed {args.seed}.', '',
              'See calibration.csv for pooled and per-fold fixed-bin calibration counts, '
              'mean probabilities, observed rates and residuals. Empty bins have missing '
              'rates. Brier and log loss are in metrics.json; lower is better. '
              'Changing bin definitions changes calibration summaries.', '',
              'The source prediction/cohort hashes and all per-fold and pooled probability '
              'metrics were checked before analysis. This verifies artifact consistency; '
              'it does not independently reconstruct training or prove source availability.', '',
              'No model performance improvement is claimed by this audit. Independent final '
              'evaluation and any calibration experiment remain future work.', '']
    artifact_path(args.output, 'report.md').write_text('\n'.join(lines))
    print(f'Wrote aggregate-only audit to {args.output}', flush=True)
    return args.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Completed src.compare_history run')
    parser.add_argument('--output', type=Path, required=True, help='New aggregate-only audit directory')
    parser.add_argument('--baseline', default='hgb_application_only')
    parser.add_argument('--candidate', default='hgb_application_bureau_installments')
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
