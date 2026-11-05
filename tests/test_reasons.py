import unittest

import numpy as np
import pandas as pd

from src.reasons import contributions, family, output_groups, reason_codes, reason_summary
from src.train import pipeline


def fitted_model(n=600, seed=0):
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError):
        raise unittest.SkipTest('Optional LightGBM/native runtime unavailable in this process.')
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    x = pd.DataFrame({'INST_LATE_PAYMENT_FRACTION': risk + rng.normal(0, .3, n),
                      'EXT_SOURCE_2': rng.uniform(size=n),
                      'AMT_CREDIT': rng.uniform(1e5, 1e6, n),
                      'NAME_CONTRACT_TYPE': rng.choice(['Cash loans', 'Revolving loans', 'Rare'], n, p=[.6, .38, .02]),
                      'NAME_TYPE_SUITE': rng.choice(['Unaccompanied', 'Family'], n)})
    x.loc[rng.choice(n, 30, replace=False), 'EXT_SOURCE_2'] = np.nan
    y = (risk + rng.normal(0, .5, n) > 1).astype(int)
    model = pipeline(x, LGBMClassifier(n_estimators=40, num_leaves=7, verbosity=-1, n_jobs=1,
                                       random_state=0)).fit(x, y)
    return model, x, y


class ReasonCodeTests(unittest.TestCase):
    def test_groups_map_every_encoded_column_to_its_input(self):
        model, x, _ = fitted_model()
        groups = output_groups(model['preprocess'])
        self.assertEqual(len(groups), len(model['preprocess'].get_feature_names_out()))
        self.assertEqual(set(groups), set(x.columns))
        # Rare categories (below min_frequency) share one indicator, not one each.
        self.assertGreaterEqual(groups.count('NAME_CONTRACT_TYPE'), 2)

    def test_contributions_add_up_to_the_model_score(self):
        model, x, _ = fitted_model()
        values, baseline = contributions(model, x)
        self.assertEqual(list(values.columns), list(x.columns))
        raw = model['model'].booster_.predict(model['preprocess'].transform(x), raw_score=True)
        np.testing.assert_allclose(values.sum(axis=1) + baseline, raw, atol=1e-6)
        probability = 1/(1+np.exp(-raw))
        np.testing.assert_allclose(probability, model.predict_proba(x)[:, 1], atol=1e-9)

    def test_reasons_are_positive_ordered_and_name_the_risk_driver(self):
        model, x, _ = fitted_model()
        codes = reason_codes(model, x, np.arange(len(x)), top_k=2)
        self.assertEqual(len(codes), len(x))
        filled = codes.loc[codes.reason_1.ne('')]
        self.assertTrue((filled.reason_1_log_odds > 0).all())
        both = filled.loc[filled.reason_2.ne('')]
        self.assertTrue((both.reason_1_log_odds >= both.reason_2_log_odds).all())
        riskiest = codes.sort_values('probability').iloc[-20:]
        self.assertEqual(riskiest.reason_1.mode()[0], 'INST_LATE_PAYMENT_FRACTION')
        self.assertEqual(riskiest.reason_1_family.mode()[0], 'Installment payment history')
        # The reported value is the applicant's own raw input for that feature.
        for row in filled.head(25).itertuples():
            raw = x.at[row.Index, row.reason_1]
            expected = 'missing' if pd.isna(raw) else (raw if isinstance(raw, str) else f'{raw:.4g}')
            self.assertEqual(row.reason_1_value, expected)
        summary = reason_summary(codes, 2)
        self.assertEqual(summary.top_reason.sum(), codes.reason_1.ne('').sum())
        self.assertTrue((summary.share_of_applicants <= 1).all())

    def test_family_labels(self):
        self.assertEqual(family('BUREAU_DEBT_SUM'), 'Credit bureau history')
        self.assertEqual(family('PREV_REFUSED_COUNT'), 'Previous Home Credit applications')
        self.assertEqual(family('EXT_SOURCE_MEAN'), 'External credit scores')
        self.assertEqual(family('AMT_CREDIT'), 'Application details')


if __name__ == '__main__':
    unittest.main()
