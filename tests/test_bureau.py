from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.bureau import (FEATURE_COLUMNS, aggregate_bureau, join_bureau,
                        read_bureau)


def bureau_fixture():
    return pd.DataFrame({
        'SK_ID_CURR': [1, 1, 2], 'SK_ID_BUREAU': [10, 11, 12],
        'DAYS_CREDIT': [-100., -20., 0.], 'DAYS_CREDIT_UPDATE': [-5., 0., 0.],
        'DAYS_ENDDATE_FACT': [np.nan, -1, np.nan],
        'CREDIT_ACTIVE': ['Active', 'Closed', 'Active'],
        'AMT_CREDIT_SUM': [100., 200., 0.],
        'AMT_CREDIT_SUM_DEBT': [40., -10., np.nan],
        'AMT_CREDIT_SUM_OVERDUE': [5., 0., np.nan],
    })


class BureauTests(unittest.TestCase):
    def test_aggregates_signed_debt_and_preserves_application_rows(self):
        raw = bureau_fixture()
        original = raw.copy(deep=True)
        aggregates, audit = aggregate_bureau(raw)
        row = aggregates.set_index('SK_ID_CURR').loc[1]
        self.assertEqual(row.BUREAU_RECORD_COUNT, 2)
        self.assertEqual(row.BUREAU_ACTIVE_FRACTION, .5)
        self.assertEqual(row.BUREAU_CREDIT_SUM, 300)
        self.assertEqual(row.BUREAU_DEBT_SUM, 30)
        self.assertAlmostEqual(row.BUREAU_DEBT_CREDIT_RATIO, .1)
        self.assertEqual(row.BUREAU_DAYS_SINCE_LAST_CREDIT, 20)
        apps = pd.DataFrame({'SK_ID_CURR': [3, 1, 2], 'TARGET': [1, 0, 1]}, index=[9, 4, 4])
        joined = join_bureau(apps, aggregates)
        pd.testing.assert_frame_equal(joined[apps.columns], apps)
        self.assertEqual(len(joined), len(apps))
        self.assertEqual(joined.iloc[0].BUREAU_RECORD_COUNT, 0)
        self.assertTrue(np.isnan(joined.iloc[0].BUREAU_CREDIT_SUM))
        self.assertTrue(np.isnan(joined.iloc[0].BUREAU_ACTIVE_FRACTION))
        self.assertEqual(joined.iloc[2].BUREAU_DAYS_SINCE_LAST_CREDIT, 0)
        self.assertEqual(audit['eligible_rows'], 3)
        pd.testing.assert_frame_equal(raw, original)

    def test_time_boundary_unknown_dates_and_future_observations(self):
        raw = pd.concat([bureau_fixture().iloc[[0]]] * 7, ignore_index=True)
        raw['SK_ID_BUREAU'] = np.arange(1, 8)
        raw.loc[0, ['DAYS_CREDIT', 'DAYS_CREDIT_UPDATE']] = 0
        raw.loc[1, 'DAYS_CREDIT_UPDATE'] = 1
        raw.loc[2, 'DAYS_CREDIT'] = 1
        raw.loc[3, 'DAYS_CREDIT_UPDATE'] = np.nan
        raw.loc[4, 'DAYS_CREDIT'] = -np.inf
        raw.loc[5, 'DAYS_ENDDATE_FACT'] = 1
        raw.loc[6, 'DAYS_ENDDATE_FACT'] = np.inf
        raw['DAYS_CREDIT_ENDDATE'] = 9999  # Scheduled future maturity is not an observed event.
        aggregates, audit = aggregate_bureau(raw)
        self.assertEqual(aggregates.iloc[0].BUREAU_RECORD_COUNT, 1)
        self.assertEqual(audit['excluded_rows'], 6)
        self.assertEqual(audit['exclusion_reasons_nonexclusive']['future_update_day'], 1)
        self.assertEqual(audit['exclusion_reasons_nonexclusive']['invalid_actual_end_day'], 1)

    def test_numerical_edges_keep_unknown_amounts_missing(self):
        raw = bureau_fixture()
        raw.loc[0, 'AMT_CREDIT_SUM_DEBT'] = np.nan
        raw.loc[1, 'AMT_CREDIT_SUM'] = 0
        raw.loc[2, 'AMT_CREDIT_SUM'] = -1
        raw.loc[2, 'AMT_CREDIT_SUM_DEBT'] = np.inf
        raw.loc[2, 'AMT_CREDIT_SUM_OVERDUE'] = -1
        raw.loc[2, 'CREDIT_ACTIVE'] = None
        aggregates, audit = aggregate_bureau(raw)
        rows = aggregates.set_index('SK_ID_CURR')
        self.assertTrue(np.isnan(rows.loc[1, 'BUREAU_DEBT_CREDIT_RATIO']))
        self.assertEqual(rows.loc[1, 'BUREAU_DEBT_SUM'], -10)
        self.assertTrue(np.isnan(rows.loc[2, 'BUREAU_CREDIT_SUM']))
        self.assertTrue(np.isnan(rows.loc[2, 'BUREAU_DEBT_SUM']))
        self.assertEqual(rows.loc[2, 'BUREAU_DEBT_KNOWN_COUNT'], 0)
        self.assertTrue(np.isnan(rows.loc[2, 'BUREAU_ACTIVE_FRACTION']))
        self.assertTrue(np.isnan(rows.loc[2, 'BUREAU_ACTIVE_COUNT']))
        self.assertFalse(np.isinf(aggregates[list(FEATURE_COLUMNS)].to_numpy()).any())
        self.assertEqual(audit['amounts_masked']['AMT_CREDIT_SUM_DEBT'], 1)

    def test_nullable_missing_identifiers_and_dates_do_not_pass_boundary(self):
        raw = bureau_fixture()
        raw['DAYS_CREDIT_UPDATE'] = pd.Series([pd.NA, 0, 0], dtype='Int64')
        aggregates, audit = aggregate_bureau(raw)
        self.assertEqual(audit['excluded_rows'], 1)
        self.assertEqual(aggregates.iloc[0].BUREAU_RECORD_COUNT, 1)
        raw['SK_ID_CURR'] = pd.Series([1, pd.NA, 2], dtype='Int64')
        with self.assertRaisesRegex(ValueError, 'nonmissing'):
            aggregate_bureau(raw)

    def test_ratio_uses_only_matching_known_balances(self):
        raw = bureau_fixture().iloc[:2].copy()
        raw.loc[1, 'AMT_CREDIT_SUM_DEBT'] = np.nan
        aggregates, _ = aggregate_bureau(raw)
        self.assertAlmostEqual(aggregates.iloc[0].BUREAU_DEBT_CREDIT_RATIO, .4)
        self.assertEqual(aggregates.iloc[0].BUREAU_RATIO_RECORD_COUNT, 1)

    def test_duplicate_ids_and_join_collisions_are_rejected(self):
        raw = bureau_fixture()
        duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'unique'):
            aggregate_bureau(duplicate)
        aggregates, _ = aggregate_bureau(raw)
        with self.assertRaisesRegex(ValueError, 'unique'):
            join_bureau(pd.DataFrame({'SK_ID_CURR': [1, 1]}), aggregates)
        with self.assertRaisesRegex(ValueError, 'unique'):
            join_bureau(pd.DataFrame({'SK_ID_CURR': [1]}), pd.concat([aggregates, aggregates]))
        with self.assertRaisesRegex(ValueError, 'collide'):
            join_bureau(pd.DataFrame({'SK_ID_CURR': [1], 'BUREAU_RECORD_COUNT': [5]}), aggregates)
        with self.assertRaisesRegex(ValueError, 'nonmissing'):
            aggregate_bureau(raw.assign(SK_ID_CURR=[1, 1, np.nan]))

    def test_no_history_or_all_excluded_has_fixed_schema(self):
        apps = pd.DataFrame({'SK_ID_CURR': [1, 7]})
        for raw in [bureau_fixture().iloc[:0], bureau_fixture().assign(DAYS_CREDIT_UPDATE=1)]:
            aggregates, _ = aggregate_bureau(raw)
            self.assertEqual(list(aggregates.columns), ['SK_ID_CURR', *FEATURE_COLUMNS])
            joined = join_bureau(apps, aggregates)
            self.assertTrue(joined.BUREAU_RECORD_COUNT.eq(0).all())
            self.assertTrue(joined.BUREAU_DEBT_CREDIT_RATIO.isna().all())

    def test_scope_and_unapproved_columns_do_not_use_other_applicants_or_labels(self):
        raw = bureau_fixture()
        expected, _ = aggregate_bureau(raw, applicant_ids=[1, 3])
        modified = raw.assign(TARGET=[1, 0, 1], FUTURE_PAYMENT=[999, -100, 2])
        modified.loc[2, 'AMT_CREDIT_SUM'] = 999999
        actual, audit = aggregate_bureau(modified, applicant_ids=[1, 3])
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(audit['scoped_rows'], 2)
        numeric, _ = aggregate_bureau(raw.assign(SK_ID_CURR=raw.SK_ID_CURR.astype(str)))
        joined = join_bureau(pd.DataFrame({'SK_ID_CURR': [1]}), numeric)
        self.assertEqual(joined.iloc[0].BUREAU_RECORD_COUNT, 2)

    def test_csv_and_zip_read_whitelist_without_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv = bureau_fixture().assign(TARGET=1).to_csv(index=False)
            (root/'bureau.csv').write_text(csv)
            with zipfile.ZipFile(root/'data.zip', 'w') as archive:
                archive.writestr('nested/bureau.csv', csv)
            plain, zipped = read_bureau(root/'bureau.csv'), read_bureau(root/'data.zip')
            pd.testing.assert_frame_equal(plain, zipped)
            self.assertNotIn('TARGET', zipped)
            self.assertFalse((root/'nested').exists())
            with zipfile.ZipFile(root/'ambiguous.zip', 'w') as archive:
                archive.writestr('a/bureau.csv', csv)
                archive.writestr('b/bureau.csv', csv)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                read_bureau(root/'ambiguous.zip')


if __name__ == '__main__':
    unittest.main()
