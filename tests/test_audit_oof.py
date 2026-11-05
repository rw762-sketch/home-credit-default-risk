import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.audit_oof import (calibration_rows, load_source, paired_bootstrap, run,
                           validate_predictions, _ranking_plan, _weighted_ranking_metrics)
from src.compare_bureau import probability_metrics, sha256


def fixture():
    return pd.DataFrame({'SK_ID_CURR': np.arange(40), 'TARGET': np.tile([0, 1], 20),
                         'fold': np.repeat([0, 1], 20),
                         'a': np.tile([.2, .8, .6, .4], 10),
                         'b': np.tile([0., 1., .4, .6], 10)})


def write_source(root):
    frame = fixture()
    assignments = frame[['SK_ID_CURR', 'TARGET', 'fold']]
    assignments.to_csv(artifact_path(root, 'fold_assignments.csv'), index=False)
    frame.to_csv(artifact_path(root, 'development_predictions.csv'), index=False)
    rows = []
    for name in ['a', 'b']:
        for fold, group in frame.groupby('fold'):
            rows.append({'variant': name, 'fold': fold, 'validation_rows': len(group),
                         'training_rows': len(frame)-len(group), 'iterations': 3, 'seconds': .1,
                         **probability_metrics(group.TARGET, group[name])})
    pd.DataFrame(rows).to_csv(artifact_path(root, 'fold_metrics.csv'), index=False)
    metrics = {'selected_development_rows': 40, 'positive_rows': 20, 'folds': 2,
               'summary': [{'variant': name, **{f'oof_{k}': v for k, v in
                           probability_metrics(frame.TARGET, frame[name]).items()}}
                           for name in ['a', 'b']]}
    artifact_path(root, 'metrics.json').write_text(json.dumps(metrics))
    manifest = {key: sha256(artifact_path(root, name)) for name, key in [
        ('fold_assignments.csv', 'fold_assignments_sha256'),
        ('development_predictions.csv', 'predictions_sha256')]}
    artifact_path(root, 'manifest.json').write_text(json.dumps(manifest))


class AuditTests(unittest.TestCase):
    def test_weighted_scores_match_explicit_sklearn_resampling_with_ties_and_zeros(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        rng = np.random.default_rng(7)
        y = np.tile([0, 1], 100)
        for p in [np.full(200, .5), rng.integers(0, 5, 200)/4, rng.uniform(size=200)]:
            plan = _ranking_plan(y, p)
            for _ in range(10):
                weights = rng.integers(0, 4, 200)
                sample = np.repeat(np.arange(200), weights)
                auc, ap = _weighted_ranking_metrics(plan, weights)
                self.assertAlmostEqual(auc, roc_auc_score(y[sample], p[sample]), places=13)
                self.assertAlmostEqual(ap, average_precision_score(y[sample], p[sample]), places=13)

    def test_rejects_invalid_predictions_and_cohorts(self):
        for column, value in [('a', np.nan), ('a', np.inf), ('a', -.1), ('a', 1.1),
                              ('TARGET', 2), ('fold', .5), ('fold', -1), ('SK_ID_CURR', None)]:
            frame = fixture()
            frame[column] = frame[column].astype(float)
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_predictions(frame)
        frame = fixture()
        frame.loc[0, 'SK_ID_CURR'] = 1
        with self.assertRaises(ValueError):
            validate_predictions(frame)
        frame = fixture()
        frame['fold'] = frame.TARGET
        with self.assertRaisesRegex(ValueError, 'both classes'):
            validate_predictions(frame)

    def test_paired_intervals_deterministic_order_invariant_and_zero_for_identical_scores(self):
        frame = fixture()
        first = paired_bootstrap(frame, 'a', 'b', 100, 7)
        self.assertEqual(first, paired_bootstrap(frame.sample(frac=1, random_state=12), 'a', 'b', 100, 7))
        for row in first:
            self.assertGreater(row['candidate_minus_baseline'], 0)
            self.assertLessEqual(row['conditional_percentile_2_5'], row['conditional_percentile_97_5'])
        for row in paired_bootstrap(frame, 'a', 'a', 100, 7):
            self.assertEqual(row['candidate_minus_baseline'], 0)
            self.assertEqual(row['conditional_percentile_2_5'], 0)
            self.assertEqual(row['conditional_percentile_97_5'], 0)

    def test_calibration_keeps_endpoints_ties_empty_bins_and_counts(self):
        frame = fixture()
        rows = pd.DataFrame(calibration_rows(frame, ['b']))
        pooled = rows[rows.split.eq('pooled OOF')]
        self.assertEqual(pooled.rows.sum(), len(frame))
        self.assertEqual(pooled.positives.sum(), frame.TARGET.sum())
        self.assertEqual(pooled.iloc[0]['rows'], 10)
        self.assertEqual(pooled.iloc[-1]['rows'], 10)
        self.assertTrue(pooled.loc[pooled.rows.eq(0), 'observed_rate'].isna().all())

    def test_source_hash_metrics_coverage_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'2026-11-05_01_source'
            source.mkdir()
            write_source(source)
            pd.testing.assert_frame_equal(load_source(source)[0], fixture())
            args = argparse.Namespace(source=source, output=root/'audit', baseline='a', candidate='b',
                                      repetitions=100, seed=42)
            run(args)
            self.assertEqual({p.name for p in args.output.iterdir()},
                             {artifact_path(args.output, name).name for name in [
                                 'report.md', 'metrics.json', 'manifest.json', 'calibration.csv', 'bootstrap_deltas.csv']})
            with self.assertRaises(FileExistsError):
                run(args)
            path = artifact_path(source, 'development_predictions.csv')
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_source(source)
            write_source(source)
            path = artifact_path(source, 'fold_metrics.csv')
            rows = pd.read_csv(path)
            rows.loc[0, 'roc_auc'] = 0
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_source(source)
            write_source(source)
            path = artifact_path(source, 'metrics.json')
            data = json.loads(path.read_text())
            data['summary'][0]['oof_brier'] += .1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'pooled metrics'):
                load_source(source)
            args.output = root/'invalid-audit'
            with self.assertRaises(ValueError):
                run(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
