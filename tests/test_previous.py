from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.previous import COUNT_COLUMNS, aggregate_previous, join_previous, read_previous


def previous_fixture():
    return pd.DataFrame({
        'SK_ID_PREV': [1, 2, 3, 4, 5], 'SK_ID_CURR': [10, 10, 10, 20, 20],
        'NAME_CONTRACT_STATUS': ['Approved', 'Refused', 'Canceled', 'Approved', 'XNA'],
        'DAYS_DECISION': [-100., -400., -50., -30., -10.],
        'AMT_APPLICATION': [100., 200., 0., 0., 50.], 'AMT_CREDIT': [120., 0., 0., 80., 50.],
        'AMT_ANNUITY': [10., np.nan, np.nan, 8., 5.], 'CNT_PAYMENT': [12., 24., np.nan, 6., 6.]})


class PreviousApplicationTests(unittest.TestCase):
    def test_status_counts_recency_and_approved_amounts(self):
        raw = previous_fixture()
        original = raw.copy(deep=True)
        aggregates, audit = aggregate_previous(raw)
        pd.testing.assert_frame_equal(raw, original)
        rows = aggregates.set_index('SK_ID_CURR')
        first = rows.loc[10]
        self.assertEqual(first.PREV_APPLICATION_COUNT, 3)
        self.assertEqual(first.PREV_APPROVED_COUNT, 1)
        self.assertEqual(first.PREV_REFUSED_COUNT, 1)
        self.assertEqual(first.PREV_CANCELED_COUNT, 1)
        self.assertAlmostEqual(first.PREV_REFUSED_FRACTION, 1/3)
        self.assertEqual(first.PREV_APPLICATION_365_COUNT, 2)
        self.assertEqual(first.PREV_REFUSED_365_COUNT, 0)
        self.assertEqual(first.PREV_DAYS_SINCE_LAST_DECISION, 50)
        self.assertEqual(first.PREV_DAYS_SINCE_LAST_APPROVAL, 100)
        self.assertEqual(first.PREV_DAYS_SINCE_LAST_REFUSAL, 400)
        self.assertEqual(first.PREV_APPROVED_CREDIT_SUM, 120)
        self.assertEqual(first.PREV_APPROVED_ANNUITY_SUM, 10)
        self.assertAlmostEqual(first.PREV_APPROVED_CREDIT_APPLICATION_RATIO, 1.2)
        self.assertEqual(first.PREV_APPROVED_CNT_PAYMENT_MEAN, 12)
        second = rows.loc[20]
        self.assertEqual(second.PREV_STATUS_KNOWN_COUNT, 1)
        self.assertEqual(second.PREV_REFUSED_FRACTION, 0)
        self.assertTrue(np.isnan(second.PREV_DAYS_SINCE_LAST_REFUSAL))
        self.assertEqual(second.PREV_APPROVED_RATIO_RECORD_COUNT, 0)
        self.assertTrue(np.isnan(second.PREV_APPROVED_CREDIT_APPLICATION_RATIO))
        self.assertEqual(audit['unknown_status_rows'], 1)

    def test_future_or_unknown_decisions_cannot_change_features(self):
        expected, _ = aggregate_previous(previous_fixture())
        extra = pd.DataFrame({
            'SK_ID_PREV': [6, 7, 8], 'SK_ID_CURR': [10, 10, 20],
            'NAME_CONTRACT_STATUS': ['Refused', 'Approved', 'Approved'],
            'DAYS_DECISION': [1., np.nan, np.inf], 'AMT_APPLICATION': [9e9, 9e9, 9e9],
            'AMT_CREDIT': [9e9, 9e9, 9e9], 'AMT_ANNUITY': [9e9, 9e9, 9e9], 'CNT_PAYMENT': [99., 99., 99.]})
        raw = pd.concat([previous_fixture(), extra], ignore_index=True)
        raw['TARGET'] = 1
        actual, audit = aggregate_previous(raw)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(audit['excluded_rows'], 3)
        self.assertEqual(audit['exclusion_reasons_nonexclusive']['future_decision_day'], 1)
        boundary = previous_fixture().iloc[[0]].assign(DAYS_DECISION=0.)
        self.assertEqual(aggregate_previous(boundary)[0].iloc[0].PREV_DAYS_SINCE_LAST_DECISION, 0)

    def test_invalid_amounts_masked_and_duplicate_records_rejected(self):
        raw = previous_fixture()
        raw.loc[0, 'AMT_CREDIT'] = -5
        aggregates, audit = aggregate_previous(raw)
        first = aggregates.set_index('SK_ID_CURR').loc[10]
        self.assertTrue(np.isnan(first.PREV_APPROVED_CREDIT_SUM))
        self.assertEqual(first.PREV_APPROVED_RATIO_RECORD_COUNT, 0)
        self.assertEqual(audit['amounts_masked']['AMT_CREDIT'], 1)
        duplicate = pd.concat([previous_fixture(), previous_fixture().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'unique'):
            aggregate_previous(duplicate)

    def test_future_duplicate_and_invalid_keys_do_not_change_past_features(self):
        raw = previous_fixture()
        future = raw.iloc[[0]].copy().assign(DAYS_DECISION=1.)
        unknown = raw.iloc[[1]].copy().assign(DAYS_DECISION=np.nan, SK_ID_CURR=np.nan)
        expected, _ = aggregate_previous(raw)
        actual, audit = aggregate_previous(pd.concat([raw, future, unknown], ignore_index=True))
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
        self.assertEqual(audit['excluded_rows'], 2)

    def test_join_preserves_rows_and_marks_absent_history(self):
        aggregates, _ = aggregate_previous(previous_fixture())
        apps = pd.DataFrame({'SK_ID_CURR': [30, 20, 10], 'TARGET': [1, 0, 1]}, index=[7, 2, 4])
        joined = join_previous(apps, aggregates)
        pd.testing.assert_frame_equal(joined[apps.columns], apps)
        self.assertEqual(joined.loc[7, list(COUNT_COLUMNS)].tolist(), [0]*len(COUNT_COLUMNS))
        self.assertTrue(np.isnan(joined.loc[7, 'PREV_APPROVED_CREDIT_SUM']))
        self.assertEqual(joined.loc[4, 'PREV_APPLICATION_COUNT'], 3)
        with self.assertRaisesRegex(ValueError, 'unique'):
            join_previous(pd.concat([apps, apps]), aggregates)
        with self.assertRaisesRegex(ValueError, 'collide'):
            join_previous(joined, aggregates)

    def test_streamed_read_scopes_ids_and_ignores_unapproved_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('previous_application.csv',
                                 previous_fixture().assign(TARGET=1).to_csv(index=False))
            frame, audit = read_previous(path, [20], chunksize=2)
            self.assertNotIn('TARGET', frame)
            self.assertEqual(frame.SK_ID_PREV.tolist(), [4, 5])
            self.assertEqual(audit['source_rows'], 5)
        empty, _ = aggregate_previous(previous_fixture().iloc[:0])
        joined = join_previous(pd.DataFrame({'SK_ID_CURR': [1]}), empty)
        self.assertEqual(joined.iloc[0].PREV_APPLICATION_COUNT, 0)


if __name__ == '__main__':
    unittest.main()
