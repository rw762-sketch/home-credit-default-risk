import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.compare_bureau import probability_metrics, run as run_bureau
from src.compare_history import (cached_bureau_run, load_checkpoint, operating_points,
                                 run, save_checkpoint)


class HistoryComparisonTests(unittest.TestCase):
    def test_checkpoint_requires_records_for_every_completed_fold(self):
        assignments = pd.DataFrame({'TARGET': np.tile([0, 1], 10),
                                    'fold': np.repeat([0, 1], 10)})
        p = np.tile([.1, .9], 10)
        row = {'variant': 'hgb_application_only', 'fold': 0, 'training_rows': 10,
               'validation_rows': 10, 'iterations': 3, 'seconds': .1,
               **probability_metrics(assignments.TARGET[:10], p[:10])}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            save_checkpoint(path, 'signature', {'hgb_application_only': p}, [row])
            with self.assertRaisesRegex(ValueError, 'unrecorded'):
                load_checkpoint(path, 'signature', assignments)
            p[10:] = np.nan
            save_checkpoint(path, 'signature', {'hgb_application_only': p}, [row])
            predictions, records = load_checkpoint(path, 'signature', assignments)
            self.assertEqual(len(records), 1)
            self.assertTrue(np.isnan(predictions['hgb_application_only'][10:]).all())

    def test_rank_ties_use_ids_and_counts_are_consistent(self):
        ids = np.arange(20, 0, -1)
        labels = (ids == 1).astype(int)
        rows = operating_points(ids, labels, np.full(20, .4))
        self.assertEqual(rows[0]['flagged'], 1)
        self.assertEqual(rows[0]['true_positive'], 1)
        self.assertEqual(rows[0]['precision'], 1)
        self.assertEqual(rows[0]['recall'], 1)
        self.assertEqual(rows[-1]['flagged'], 0)
        self.assertEqual(rows, operating_points(ids[::-1], labels[::-1], np.full(20, .4)))

    def test_staged_bundle_and_verified_cache_preserve_cohort_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = pd.DataFrame({'SK_ID_CURR': np.arange(1, 201),
                                  'TARGET': np.tile([0, 0, 0, 1], 50), 'amount': np.arange(200.)})
            train = root/'train.csv'
            frame.to_csv(train, index=False)
            bureau = pd.DataFrame({'SK_ID_CURR': np.arange(1, 151), 'SK_ID_BUREAU': np.arange(1001, 1151),
                                   'DAYS_CREDIT': -100, 'DAYS_CREDIT_UPDATE': -2, 'CREDIT_ACTIVE': 'Active',
                                   'AMT_CREDIT_SUM': 100., 'AMT_CREDIT_SUM_DEBT': 20., 'AMT_CREDIT_SUM_OVERDUE': 0.})
            payments = pd.DataFrame({'SK_ID_CURR': np.arange(1, 151), 'SK_ID_PREV': np.arange(301, 451),
                                     'NUM_INSTALMENT_NUMBER': 1, 'NUM_INSTALMENT_VERSION': 1,
                                     'DAYS_INSTALMENT': -10., 'DAYS_ENTRY_PAYMENT': -5.,
                                     'AMT_INSTALMENT': 50., 'AMT_PAYMENT': 20.})
            archive = root/'data.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('bureau.csv', bureau.to_csv(index=False))
                z.writestr('installments_payments.csv', payments.to_csv(index=False))
            baseline_args = argparse.Namespace(train=train, bureau=archive, output=root/'2026-10-14_01_bureau',
                                               folds=2, max_development_rows=100, max_iter=3, threads=1)
            args = argparse.Namespace(train=train, archive=archive, reuse_bureau_run=baseline_args.output,
                                      output=root/'2026-10-14_02_history', folds=2, max_development_rows=100,
                                      max_iter=3, lgbm_estimators=3, threads=1, include_lightgbm=False)
            with redirect_stdout(StringIO()):
                run_bureau(baseline_args)
                out = run(args)
            summary = json.loads(artifact_path(out, 'metrics.json').read_text())
            self.assertEqual(summary['selected_development_rows'], 100)
            self.assertEqual(summary['excluded_holdout_rows'], 40)
            folds = pd.read_csv(artifact_path(out, 'fold_metrics.csv'))
            self.assertEqual(int(folds.cached.sum()), 4)
            points = pd.read_csv(artifact_path(out, 'operating_points.csv'))
            self.assertEqual(len(points[points.split.eq('pooled OOF')]), 12)
            paired = pd.read_csv(artifact_path(out, 'paired_deltas.csv'))
            self.assertEqual(len(paired), 4)
            assignments = pd.read_csv(artifact_path(out, 'fold_assignments.csv'))
            cached_folds_path = artifact_path(baseline_args.output, 'fold_metrics.csv')
            original_folds = pd.read_csv(cached_folds_path)
            duplicate = original_folds.copy()
            duplicate.iloc[1] = duplicate.iloc[0]
            duplicate.to_csv(cached_folds_path, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                cached_bureau_run(baseline_args.output, assignments, train, archive, 3)
            inconsistent = original_folds.copy()
            inconsistent.loc[0, 'roc_auc'] += .1
            inconsistent.to_csv(cached_folds_path, index=False)
            with self.assertRaisesRegex(ValueError, 'disagree'):
                cached_bureau_run(baseline_args.output, assignments, train, archive, 3)
            original_folds.to_csv(cached_folds_path, index=False)
            assignments.loc[0, 'TARGET'] = 1-assignments.loc[0, 'TARGET']
            with self.assertRaises(AssertionError):
                cached_bureau_run(baseline_args.output, assignments, train, archive, 3)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                cached_bureau_run(baseline_args.output, assignments, train, archive, 4)
            self.assertFalse(any(out.glob('*holdout*')))
            self.assertFalse(any(out.glob('*.joblib')))
            args.resume = True
            with redirect_stdout(StringIO()), patch('src.compare_history.pipeline', side_effect=AssertionError('Duplicate fit')):
                run(args)
            args.lgbm_estimators = 4
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'Checkpoint'):
                run(args)

    def test_lightgbm_available_and_uses_explicit_bounded_threads(self):
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError):
            self.skipTest('Optional LightGBM/native runtime unavailable in this process.')
        x = np.arange(80.).reshape(-1, 1)
        model = LGBMClassifier(n_estimators=3, n_jobs=1, verbosity=-1,
                              deterministic=True, force_col_wise=True).fit(x, np.tile([0, 1], 40))
        self.assertEqual(model.get_params()['n_jobs'], 1)
        self.assertTrue(np.isfinite(model.predict_proba(x)).all())


if __name__ == '__main__':
    unittest.main()
