import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.submit import prepare, run


def write_fixture(root, n_train=240, n_test=60):
    rng = np.random.default_rng(0)
    ids = np.arange(1, n_train + n_test + 1)
    signal = rng.normal(size=len(ids))
    apps = pd.DataFrame({'SK_ID_CURR': ids, 'AMT_CREDIT': rng.uniform(1e5, 1e6, len(ids)),
                         'AMT_INCOME_TOTAL': rng.uniform(5e4, 3e5, len(ids)),
                         'AMT_ANNUITY': rng.uniform(5e3, 5e4, len(ids)),
                         'EXT_SOURCE_2': rng.uniform(size=len(ids)),
                         'NAME_CONTRACT_TYPE': rng.choice(['Cash loans', 'Revolving loans'], len(ids))})
    train = apps.iloc[:n_train].assign(TARGET=(signal[:n_train] > 1).astype(int))
    test = apps.iloc[n_train:]
    train.to_csv(root/'train.csv', index=False)
    test.to_csv(root/'test.csv', index=False)
    bureau = pd.DataFrame({'SK_ID_CURR': ids[::2], 'SK_ID_BUREAU': np.arange(1, len(ids[::2])+1),
                           'DAYS_CREDIT': -100, 'DAYS_CREDIT_UPDATE': -2, 'CREDIT_ACTIVE': 'Active',
                           'AMT_CREDIT_SUM': 100., 'AMT_CREDIT_SUM_DEBT': 20., 'AMT_CREDIT_SUM_OVERDUE': 0.})
    payments = pd.DataFrame({'SK_ID_CURR': ids, 'SK_ID_PREV': ids + 10_000, 'NUM_INSTALMENT_NUMBER': 1,
                             'NUM_INSTALMENT_VERSION': 1, 'DAYS_INSTALMENT': -10.,
                             'DAYS_ENTRY_PAYMENT': -10. + np.clip(signal * 5, 0, None),
                             'AMT_INSTALMENT': 50., 'AMT_PAYMENT': 50.})
    previous = pd.DataFrame({'SK_ID_PREV': ids + 20_000, 'SK_ID_CURR': ids,
                             'NAME_CONTRACT_STATUS': np.where(signal > .5, 'Refused', 'Approved'),
                             'DAYS_DECISION': -30., 'AMT_APPLICATION': 100., 'AMT_CREDIT': 90.,
                             'AMT_ANNUITY': 9., 'CNT_PAYMENT': 12.})
    with zipfile.ZipFile(root/'data.zip', 'w') as z:
        z.writestr('bureau.csv', bureau.to_csv(index=False))
        z.writestr('installments_payments.csv', payments.to_csv(index=False))
        z.writestr('previous_application.csv', previous.to_csv(index=False))
    return train, test


class SubmissionTests(unittest.TestCase):
    def test_prepare_aligns_features_and_rejects_overlap_or_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, test = write_fixture(root)
            x_train, y, x_test, test_ids, audits = prepare(train, test, root/'data.zip')
            self.assertEqual(list(x_train.columns), list(x_test.columns))
            self.assertEqual((len(x_train), len(x_test)), (len(train), len(test)))
            self.assertEqual(list(test_ids), test.SK_ID_CURR.tolist())
            self.assertNotIn('TARGET', x_test)
            for column in ['BUREAU_RECORD_COUNT', 'INST_PAYMENT_DELAY_MEAN', 'EXT_SOURCE_MEAN', 'PREV_REFUSED_COUNT']:
                self.assertIn(column, x_train)
            self.assertEqual(set(audits), {'bureau', 'installments', 'previous'})
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare(train, train.drop(columns='TARGET'), root/'data.zip')
            with self.assertRaisesRegex(ValueError, 'TARGET'):
                prepare(train, test.assign(TARGET=0), root/'data.zip')

    def test_run_writes_submission_model_and_reasons_without_uploading(self):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest('Optional LightGBM/native runtime unavailable in this process.')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, test = write_fixture(root)
            args = argparse.Namespace(train=root/'train.csv', test=root/'test.csv', archive=root/'data.zip',
                                      algorithm='lgbm', learning_rate=.1, max_rounds=200, patience=5,
                                      threads=1, reasons=2, reason_cutoff=None,
                                      output=root/'2026-11-05_01_submission')
            with redirect_stdout(StringIO()):
                out = run(args)
            submission = pd.read_csv(artifact_path(out, 'submission.csv'))
            self.assertEqual(list(submission.columns), ['SK_ID_CURR', 'TARGET'])
            self.assertEqual(submission.SK_ID_CURR.tolist(), test.SK_ID_CURR.tolist())
            self.assertTrue(submission.TARGET.between(0, 1).all())
            metrics = json.loads(artifact_path(out, 'metrics.json').read_text())
            self.assertIsNone(metrics['kaggle_score'])
            self.assertLessEqual(metrics['selected_iterations'], 200)
            codes = pd.read_csv(artifact_path(out, 'reason_codes.csv'), keep_default_na=False)
            self.assertEqual(len(codes), len(test))
            np.testing.assert_allclose(codes.probability, submission.TARGET, atol=1e-12)
            self.assertTrue(artifact_path(out, 'submission_model.joblib').exists())
            self.assertIn('not** been uploaded', artifact_path(out, 'report.md').read_text())
            args.output, args.reason_cutoff = root/'2026-11-05_02_submission', 1.1
            with redirect_stdout(StringIO()):
                out = run(args)
            self.assertEqual(len(pd.read_csv(artifact_path(out, 'reason_codes.csv'))), 0)


if __name__ == '__main__':
    unittest.main()
