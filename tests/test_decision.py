import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.decision import (compare_policies, cost_cutoff, loss_by_band, policy_outcome,
                          profit_curve, run, sensitivity, top_k_reject)


def fixture(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, .4, n)
    y = rng.binomial(1, p)
    ead = rng.uniform(1e5, 1e6, n)
    return y, p, ead, np.arange(1, n+1)


class DecisionTests(unittest.TestCase):
    def test_cost_cutoff_is_break_even_probability(self):
        t = cost_cutoff(.1, .45)
        self.assertAlmostEqual(t, .1/.55)
        self.assertAlmostEqual((1-t)*.1 - t*.45, 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            cost_cutoff(0, .45)

    def test_profit_hand_example(self):
        y = np.array([0, 1, 0])
        p = np.array([.1, .2, .9])
        ead = np.array([100., 200., 300.])
        row = policy_outcome('x', p < .5, y, p, ead, .1, .5)
        self.assertAlmostEqual(row['realized_profit'], .1*100 - .5*200)
        self.assertAlmostEqual(row['expected_profit'], (.9*.1 - .1*.5)*100 + (.8*.1 - .2*.5)*200)
        self.assertEqual((row['approved'], row['defaults_caught'], row['good_rejected']), (2, 0, 1))
        self.assertAlmostEqual(row['bad_rate_approved'], .5)

    def test_profit_curve_matches_direct_policy_evaluation(self):
        y, p, ead, _ = fixture()
        curve = profit_curve(y, p, ead, .1, .45, points=25)
        for row in curve.sample(8, random_state=1).itertuples():
            direct = policy_outcome('t', p < row.threshold, y, p, ead, .1, .45)
            self.assertAlmostEqual(row.realized_profit, direct['realized_profit'], places=4)
            self.assertAlmostEqual(row.expected_profit, direct['expected_profit'], places=4)
        self.assertEqual(curve.approval_rate.iloc[-1], 1)
        self.assertEqual(curve.approval_rate.iloc[0], 0)

    def test_top_k_rejects_exact_count_with_id_tie_break(self):
        p = np.full(10, .3)
        ids = np.arange(10, 0, -1)
        approve = top_k_reject(p, ids, .2)
        self.assertEqual((~approve).sum(), 2)
        self.assertEqual(set(ids[~approve]), {1, 2})

    def test_oracle_bounds_every_threshold_policy(self):
        y, p, ead, ids = fixture()
        policies, _ = compare_policies(y, p, ead, ids, .1, .45, f2_cutoff=.1)
        oracle = policies.set_index('policy').loc['best observed cutoff (oracle)', 'realized_profit']
        threshold_rows = policies[policies.policy.str.contains('cutoff|PD|approve all')]
        self.assertTrue((threshold_rows.realized_profit <= oracle + 1e-6).all())
        self.assertEqual(len(policies), 8)

    def test_sensitivity_and_bands(self):
        y, p, ead, _ = fixture()
        grid = sensitivity(y, p, ead, [.05, .2], [.3, .6])
        self.assertEqual(len(grid), 4)
        # A higher margin or a lower LGD means approving more applicants.
        cut = grid.set_index(['margin', 'lgd']).cost_cutoff
        self.assertGreater(cut[(.2, .3)], cut[(.05, .3)])
        self.assertGreater(cut[(.05, .3)], cut[(.05, .6)])
        self.assertTrue((grid.profit_cost_based <= grid.profit_oracle + 1e-6).all())
        bands = loss_by_band(y, p, ead, .45)
        self.assertEqual(bands.applicants.sum(), len(y))
        self.assertAlmostEqual(bands.expected_loss.sum(), (.45*p*ead).sum(), places=3)
        self.assertTrue(bands.mean_pd.is_monotonic_increasing)

    def test_run_writes_reports_from_saved_predictions(self):
        y, p, ead, ids = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root/'predictions.csv'
            pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y, 'fold': ids % 5, 'model_a': p}).to_csv(predictions, index=False)
            train = root/'train.csv'
            pd.DataFrame({'SK_ID_CURR': ids[::-1], 'AMT_CREDIT': ead[::-1]}).to_csv(train, index=False)
            args = argparse.Namespace(source=None, predictions=predictions, variant='model_a', train=train,
                                      margin=.1, lgd=.45, margins=[.1], lgds=[.45],
                                      output=root/'2026-11-05_01_decision')
            with redirect_stdout(StringIO()):
                out = run(args)
            metrics = json.loads(artifact_path(out, 'metrics.json').read_text())
            self.assertAlmostEqual(metrics['cost_cutoff'], .1/.55)
            self.assertAlmostEqual(metrics['expected_vs_realized_loss_all']['realized'], (.45*y*ead).sum(), places=2)
            self.assertTrue(artifact_path(out, 'report.md').exists())
            args.variant, args.output = 'missing', root/'2026-11-05_02_decision'
            with self.assertRaisesRegex(ValueError, 'model_a'):
                run(args)


if __name__ == '__main__':
    unittest.main()
