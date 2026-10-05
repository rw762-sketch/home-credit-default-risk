import unittest
import numpy as np
from sklearn.metrics import fbeta_score
from src.monitoring import threshold_sweep, psi, calibration_table


class MonitoringTests(unittest.TestCase):
    def test_threshold_uses_inclusive_ties_and_maximizes_f2(self):
        y=np.array([0,1,0,1,0,1]);p=np.array([.1,.2,.2,.4,.6,.6])
        table,chosen=threshold_sweep(y,p)
        for row in table.itertuples():
            self.assertAlmostEqual(row.f2,fbeta_score(y,p>=row.threshold,beta=2))
        self.assertAlmostEqual(fbeta_score(y,p>=chosen,beta=2),table.f2.max())

    def test_psi_handles_unseen_categories_missing_and_outliers(self):
        result=psi(['A','A','B',None],['A','NEW',None,None])
        self.assertTrue(np.isfinite(result['psi']))
        self.assertGreater(result['psi'],0)
        reference=np.arange(100,dtype=float)
        self.assertAlmostEqual(psi(reference,reference)['psi'],0.)
        shifted=psi(reference,reference+10000)
        self.assertTrue(shifted['exceeds_null_p95'])

    def test_calibration_preserves_counts_with_tied_scores(self):
        y=np.array([0,1,0,1,1,0]);p=np.array([.1,.1,.2,.2,.9,.9])
        table=calibration_table(y,p,bins=3)
        self.assertEqual(table['count'].sum(),len(y))
        self.assertAlmostEqual(np.average(table.observed_rate,weights=table['count']),y.mean())
        constant=calibration_table(y,np.full(len(y),.1))
        self.assertEqual(constant['count'].sum(),len(y))
        self.assertEqual(len(constant),1)


if __name__=='__main__':
    unittest.main()
