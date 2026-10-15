from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.installments import aggregate_installments, join_installments, read_installments


def payment_fixture():
    return pd.DataFrame({
        'SK_ID_CURR': [1, 1, 2], 'SK_ID_PREV': [10, 10, 20],
        'NUM_INSTALMENT_NUMBER': [1, 1, 1], 'NUM_INSTALMENT_VERSION': [1, 1, 1],
        'DAYS_INSTALMENT': [-25., -25., -10.], 'DAYS_ENTRY_PAYMENT': [-30., -20., -5.],
        'AMT_INSTALMENT': [100., 100., 50.], 'AMT_PAYMENT': [40., 60., 20.]})


class InstallmentTests(unittest.TestCase):
    def test_partial_tranches_count_one_obligation_and_one_due_amount(self):
        raw = payment_fixture()
        original = raw.copy(deep=True)
        aggregates, audit = aggregate_installments(raw)
        row = aggregates.set_index('SK_ID_CURR').loc[1]
        self.assertEqual(row.INST_EVENT_COUNT, 2)
        self.assertEqual(row.INST_OBLIGATION_COUNT, 1)
        self.assertEqual(row.INST_PAIRED_DUE_SUM, 100)
        self.assertEqual(row.INST_PAIRED_PAYMENT_SUM, 100)
        self.assertEqual(row.INST_SHORTFALL_SUM, 0)
        self.assertEqual(row.INST_PAYMENT_DELAY_MEAN, 2.5)
        self.assertEqual(row.INST_LATE_PAYMENT_FRACTION, .5)
        second = aggregates.set_index('SK_ID_CURR').loc[2]
        self.assertEqual(second.INST_SHORTFALL_SUM, 30)
        self.assertAlmostEqual(second.INST_SHORTFALL_RATIO, .6)
        self.assertEqual(audit['eligible_obligations'], 2)
        pd.testing.assert_frame_equal(raw, original)

    def test_future_payment_cannot_change_past_amounts_or_version_selection(self):
        raw = payment_fixture()
        raw.loc[1, 'DAYS_ENTRY_PAYMENT'] = 1
        expected, _ = aggregate_installments(raw)
        row = expected.set_index('SK_ID_CURR').loc[1]
        self.assertEqual(row.INST_SHORTFALL_SUM, 60)
        changed = raw.copy()
        changed.loc[1, ['AMT_INSTALMENT', 'AMT_PAYMENT', 'NUM_INSTALMENT_VERSION']] = [9999, 8888, 3]
        changed['TARGET'] = [1, 0, 1]
        actual, audit = aggregate_installments(changed)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(audit['exclusion_reasons_nonexclusive']['future_payment_date'], 1)

    def test_unknown_future_due_and_day_zero_boundaries(self):
        raw = pd.concat([payment_fixture().iloc[[2]]] * 5, ignore_index=True)
        raw['SK_ID_PREV'] = np.arange(1, 6)
        raw.loc[0, ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']] = 0
        raw.loc[1, 'DAYS_INSTALMENT'] = 1
        raw.loc[2, 'DAYS_ENTRY_PAYMENT'] = np.nan
        raw.loc[3, 'DAYS_INSTALMENT'] = -np.inf
        raw.loc[4, 'NUM_INSTALMENT_VERSION'] = np.nan
        aggregates, audit = aggregate_installments(raw)
        self.assertEqual(aggregates.iloc[0].INST_EVENT_COUNT, 1)
        self.assertEqual(aggregates.iloc[0].INST_DAYS_SINCE_LAST_PAYMENT, 0)
        self.assertEqual(audit['time_or_key_excluded_rows'], 4)

    def test_ambiguous_versions_schedules_and_duplicate_events_are_excluded(self):
        for field, value in [('NUM_INSTALMENT_VERSION', 2), ('AMT_INSTALMENT', 101),
                              ('DAYS_INSTALMENT', -26)]:
            with self.subTest(field=field):
                raw = payment_fixture()
                raw.loc[1, field] = value
                aggregates, audit = aggregate_installments(raw)
                self.assertEqual(aggregates.SK_ID_CURR.tolist(), [2])
                self.assertEqual(audit['ambiguous_schedule_or_duplicate_rows_excluded'], 2)
        duplicate = pd.concat([payment_fixture(), payment_fixture().iloc[[0]]], ignore_index=True)
        aggregates, _ = aggregate_installments(duplicate)
        self.assertEqual(aggregates.SK_ID_CURR.tolist(), [2])

    def test_numeric_unknown_tranches_zero_due_overpayments_and_missing_histories(self):
        raw = payment_fixture()
        raw.loc[1, 'AMT_PAYMENT'] = np.inf
        raw.loc[2, 'AMT_INSTALMENT'] = 0
        aggregates, _ = aggregate_installments(raw)
        rows = aggregates.set_index('SK_ID_CURR')
        self.assertTrue(np.isnan(rows.loc[1, 'INST_SHORTFALL_SUM']))
        self.assertEqual(rows.loc[1, 'INST_NUMERIC_OBLIGATION_COUNT'], 0)
        self.assertEqual(rows.loc[2, 'INST_SHORTFALL_SUM'], 0)
        self.assertTrue(np.isnan(rows.loc[2, 'INST_SHORTFALL_RATIO']))
        apps = pd.DataFrame({'SK_ID_CURR': [3, 2, 1], 'TARGET': [1, 0, 1]}, index=[7, 2, 2])
        joined = join_installments(apps, aggregates)
        pd.testing.assert_frame_equal(joined[apps.columns], apps)
        self.assertEqual(len(joined), 3)
        self.assertEqual(joined.iloc[0].INST_EVENT_COUNT, 0)
        self.assertTrue(np.isnan(joined.iloc[0].INST_SHORTFALL_SUM))
        self.assertFalse(np.isinf(joined.select_dtypes(include='number').to_numpy()).any())
        with self.assertRaisesRegex(ValueError, 'unique'):
            join_installments(pd.concat([apps, apps]), aggregates)
        with self.assertRaisesRegex(ValueError, 'unique'):
            join_installments(apps, pd.concat([aggregates, aggregates]))

    def test_chunk_boundaries_do_not_split_obligations_or_read_unapproved_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('installments_payments.csv', payment_fixture().assign(TARGET=1).to_csv(index=False))
            frame, audit = read_installments(path, [1], chunksize=1)
            self.assertNotIn('TARGET', frame)
            result, _ = aggregate_installments(frame)
            self.assertEqual(result.iloc[0].INST_PAIRED_DUE_SUM, 100)
            self.assertEqual(audit['source_rows'], 3)
            self.assertEqual(audit['scoped_rows'], 2)
        empty, _ = aggregate_installments(payment_fixture().iloc[:0])
        joined = join_installments(pd.DataFrame({'SK_ID_CURR': [1]}), empty)
        self.assertEqual(joined.iloc[0].INST_EVENT_COUNT, 0)


if __name__ == '__main__':
    unittest.main()
