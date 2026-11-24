"""Missing lag semantics in the production feature builder."""
import unittest
import numpy as np
import test_feature_math
from auto_engine import Features


class MissingLagTests(unittest.TestCase):
    def test_unavailable_lag_is_not_observed_zero(self):
        for frequency,t in [('monthly',11),('bimonthly',5)]:
            for metric in ('volume_kg','value_1000_toman'):
                with self.subTest(frequency=frequency,metric=metric):
                    d=test_feature_math.FeatureMathTests().panel(frequency)
                    y=d.metric(metric)
                    y[0,0]=0
                    y[1,0]=.4
                    y[1,t]=.7
                    x,_,_=Features(d,metric,0,23).features(t,1)
                    for lag in (1,2,3,5,11,12):
                        values=x[f'sales_minus_{lag}'].to_numpy()
                        if lag>t:
                            self.assertTrue(np.isnan(values).all())
                        else:
                            np.testing.assert_array_equal(values,y[:,t-lag])
                    # At the next relevant origin lag12 is a true observed
                    # zero for A, and a true positive .4 for B.
                    z,_,_=Features(d,metric,0,23).features(12,1)
                    self.assertEqual(z.loc[0,'sales_minus_12'],0.)
                    self.assertEqual(z.loc[1,'sales_minus_12'],.4)

    def test_training_window_cannot_read_before_its_start(self):
        d=test_feature_math.FeatureMathTests().panel()
        d.volume[:,0]=999
        x,_,_=Features(d,'volume_kg',1,23).features(11,3)
        self.assertTrue(np.isnan(x['sales_minus_12']).all())
        np.testing.assert_array_equal(x['sales_minus_11'],d.volume[:,1])


if __name__=='__main__':
    unittest.main()
