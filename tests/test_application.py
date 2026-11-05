import unittest

import numpy as np
import pandas as pd

from src.application import FEATURE_COLUMNS, add_application_features
from src.train import features


def application_fixture():
    return pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3], 'TARGET': [0, 1, 0],
        'EXT_SOURCE_1': [.2, np.nan, np.nan], 'EXT_SOURCE_2': [.4, .5, np.nan],
        'EXT_SOURCE_3': [.6, .7, np.nan],
        'DAYS_EMPLOYED': [-1000, 365243, -10], 'DAYS_BIRTH': [-10000, -20000, -5000],
        'AMT_GOODS_PRICE': [90., 100., 1.], 'AMT_CREDIT': [100., 0., 2.],
        'AMT_INCOME_TOTAL': [300., 100., 50.], 'CNT_FAM_MEMBERS': [3., 0., np.nan]},
        index=[5, 3, 9])


class ApplicationFeatureTests(unittest.TestCase):
    def test_external_score_summaries_use_known_sources_only(self):
        raw = application_fixture()
        original = raw.copy(deep=True)
        result = add_application_features(raw)
        pd.testing.assert_frame_equal(raw, original)
        pd.testing.assert_index_equal(result.index, raw.index)
        self.assertEqual(list(result.columns[-len(FEATURE_COLUMNS):]), list(FEATURE_COLUMNS))
        full, partial, empty = result.iloc[0], result.iloc[1], result.iloc[2]
        self.assertAlmostEqual(full.EXT_SOURCE_MEAN, .4)
        self.assertAlmostEqual(full.EXT_SOURCE_MIN, .2)
        self.assertAlmostEqual(full.EXT_SOURCE_MAX, .6)
        self.assertAlmostEqual(full.EXT_SOURCE_STD, np.sqrt(.08/3))
        self.assertAlmostEqual(full.EXT_SOURCE_PRODUCT, .048)
        self.assertEqual(full.EXT_SOURCE_KNOWN_COUNT, 3)
        self.assertAlmostEqual(partial.EXT_SOURCE_MEAN, .6)
        self.assertAlmostEqual(partial.EXT_SOURCE_STD, .1)
        self.assertTrue(np.isnan(partial.EXT_SOURCE_PRODUCT))
        self.assertEqual(empty.EXT_SOURCE_KNOWN_COUNT, 0)
        for column in ['EXT_SOURCE_MEAN', 'EXT_SOURCE_STD', 'EXT_SOURCE_PRODUCT']:
            self.assertTrue(np.isnan(empty[column]))

    def test_ratios_handle_sentinel_zero_and_missing_denominators(self):
        result = add_application_features(application_fixture())
        self.assertAlmostEqual(result.iloc[0].EMPLOYED_BIRTH_RATIO, .1)
        self.assertTrue(np.isnan(result.iloc[1].EMPLOYED_BIRTH_RATIO))
        self.assertAlmostEqual(result.iloc[0].GOODS_CREDIT_RATIO, .9)
        self.assertTrue(np.isnan(result.iloc[1].GOODS_CREDIT_RATIO))
        self.assertAlmostEqual(result.iloc[0].INCOME_PER_FAMILY_MEMBER, 100)
        self.assertTrue(np.isnan(result.iloc[1].INCOME_PER_FAMILY_MEMBER))
        self.assertTrue(np.isnan(result.iloc[2].INCOME_PER_FAMILY_MEMBER))
        x = features(result)
        self.assertNotIn('TARGET', x)
        self.assertFalse(np.isinf(x.select_dtypes(include='number').to_numpy()).any())

    def test_missing_source_columns_give_missing_features_and_collisions_fail(self):
        result = add_application_features(pd.DataFrame({'SK_ID_CURR': [1, 2]}))
        self.assertTrue(result.EXT_SOURCE_MEAN.isna().all())
        self.assertEqual(result.EXT_SOURCE_KNOWN_COUNT.tolist(), [0, 0])
        with self.assertRaisesRegex(ValueError, 'collide'):
            add_application_features(result)

    def test_labels_cannot_change_features(self):
        raw = application_fixture()
        flipped = raw.assign(TARGET=1 - raw.TARGET)
        columns = list(FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(add_application_features(raw)[columns],
                                      add_application_features(flipped)[columns])


if __name__ == '__main__':
    unittest.main()
