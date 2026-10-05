import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.train import features, pipeline, validate


class PipelineTests(unittest.TestCase):
    def test_ids_target_and_invalid_ratios(self):
        frame = pd.DataFrame({'SK_ID_CURR': [1, 2], 'TARGET': [0, 1],
            'AMT_CREDIT': [10., 20.], 'AMT_INCOME_TOTAL': [0., 5.],
            'DAYS_EMPLOYED': [365243, -20]})
        x = features(frame)
        self.assertNotIn('TARGET', x)
        self.assertNotIn('SK_ID_CURR', x)
        self.assertTrue(np.isnan(x.loc[0, 'CREDIT_INCOME_RATIO']))
        self.assertTrue(np.isnan(x.loc[0, 'DAYS_EMPLOYED']))

    def test_preprocessing_uses_training_data_and_handles_new_category(self):
        x = pd.DataFrame({'amount': [1., 3., np.nan, 5.],
                          'category': ['A', 'A', 'B', 'B']})
        model = pipeline(x, LogisticRegression()).fit(x, [0, 0, 1, 1])
        imputer = model['preprocess'].named_transformers_['numeric']['impute']
        self.assertEqual(imputer.statistics_[0], 3.)
        result = model.predict_proba(pd.DataFrame({'amount': [9999., np.nan],
                                                  'category': ['NEW', None]}))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(imputer.statistics_[0], 3.)

    def test_duplicate_applicants_rejected(self):
        frame = pd.DataFrame({'SK_ID_CURR': [1]*10, 'TARGET': [0, 1]*5})
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate(frame, 2)


if __name__ == '__main__':
    unittest.main()
