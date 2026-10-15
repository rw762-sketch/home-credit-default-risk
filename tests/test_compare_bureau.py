import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.artifacts import artifact_path
from src.compare_bureau import development_folds, matched_predictions, run
from src.train import SEED


def applications(n=200):
    return pd.DataFrame({'SK_ID_CURR': np.arange(1, n+1),
                         'TARGET': np.tile([0, 0, 0, 1], n//4),
                         'amount': np.arange(n, dtype=float),
                         'category': np.tile(['A', 'B'], n//2)})


class BureauComparisonTests(unittest.TestCase):
    def test_saved_protocol_folds_reused_and_holdout_excluded_after_cap(self):
        frame = applications()
        selected, folds, ndev, nhold = development_folds(frame, folds=5, max_rows=100)
        dev, hold = train_test_split(np.arange(len(frame)), test_size=.2,
                                    stratify=frame.TARGET, random_state=SEED)
        original = {}
        for fold, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
                dev, frame.TARGET.iloc[dev])):
            original.update({int(frame.SK_ID_CURR.iloc[dev[i]]): fold for i in val})
        self.assertEqual((ndev, nhold), (160, 40))
        self.assertFalse(set(selected.SK_ID_CURR) & set(frame.SK_ID_CURR.iloc[hold]))
        self.assertEqual(folds.tolist(), [original[int(i)] for i in selected.SK_ID_CURR])
        again, again_folds, _, _ = development_folds(frame, folds=5, max_rows=100)
        pd.testing.assert_frame_equal(again, selected)
        np.testing.assert_array_equal(again_folds, folds)
        full, _, _, _ = development_folds(frame, folds=5, max_rows=0)
        self.assertEqual(len(full), 160)
        with self.assertRaises(ValueError):
            development_folds(frame, max_rows=4)

    def test_identical_feature_variants_receive_identical_oof_predictions(self):
        frame = applications(80)
        x = frame[['amount', 'category']]
        fold_ids = np.arange(len(x)) % 3
        with redirect_stdout(StringIO()):
            predictions, table = matched_predictions(
                {'first': x, 'same': x.copy()}, frame.TARGET, fold_ids, max_iter=3, threads=1)
        np.testing.assert_array_equal(predictions['first'], predictions['same'])
        self.assertTrue(np.isfinite(predictions['first']).all())
        self.assertEqual(len(table), 6)
        with self.assertRaisesRegex(ValueError, 'indices'):
            matched_predictions({'first': x, 'bad_order': x.iloc[::-1]}, frame.TARGET, fold_ids)

    def test_cli_workflow_writes_auditable_bundle_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = applications()
            train = root/'train.csv'
            frame.to_csv(train, index=False)
            bureau = pd.DataFrame({
                'SK_ID_CURR': np.arange(1, 151), 'SK_ID_BUREAU': np.arange(1001, 1151),
                'DAYS_CREDIT': -10, 'DAYS_CREDIT_UPDATE': -1,
                'CREDIT_ACTIVE': 'Active', 'AMT_CREDIT_SUM': 100,
                'AMT_CREDIT_SUM_DEBT': 20, 'AMT_CREDIT_SUM_OVERDUE': 0})
            source = root/'bureau.csv'
            bureau.to_csv(source, index=False)
            args = argparse.Namespace(train=train, bureau=source, output=root/'run',
                                      folds=2, max_development_rows=100, max_iter=3, threads=1)
            with redirect_stdout(StringIO()):
                out = run(args)
            metrics = json.loads(artifact_path(out, 'metrics.json').read_text())
            manifest = json.loads(artifact_path(out, 'manifest.json').read_text())
            audit = json.loads(artifact_path(out, 'bureau_audit.json').read_text())
            self.assertEqual(metrics['selected_development_rows'], 100)
            self.assertEqual(metrics['excluded_holdout_rows'], 40)
            self.assertEqual(len(metrics['paired_fold_deltas']), 2)
            self.assertEqual(len(manifest['inputs']), 2)
            self.assertGreater(audit['selected_applicants_without_eligible_history'], 0)
            pred = pd.read_csv(artifact_path(out, 'development_predictions.csv'))
            self.assertEqual(len(pred), 100)
            self.assertEqual(set(pred.columns), {'SK_ID_CURR', 'TARGET', 'fold',
                                                  'application_only', 'application_plus_bureau'})
            self.assertFalse(any(out.glob('*holdout*')))
            self.assertFalse(any(out.glob('*.joblib')))
            self.assertIn('Development evidence only', artifact_path(out, 'report.md').read_text())
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(train.read_text(), frame.to_csv(index=False))


if __name__ == '__main__':
    unittest.main()
