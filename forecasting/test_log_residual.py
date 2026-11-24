"""Independent arithmetic and production-path regression checks for Log Residual."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import auto_engine
from app_core import Dataset, DIMS
from auto_engine import log_reference_scale, log_residual_target, log_residual_prediction


class LogResidualTests(unittest.TestCase):
    def test_reference_uses_last_positive_not_peak_or_future(self):
        history=np.array([[500.,100.,0.,0.],[0.,0.,0.,0.],[1.,2.,3.,4.]])
        np.testing.assert_array_equal(log_reference_scale(history,np.array([25.,0.,8.])),
                                      [100.,0.,8.])

    def test_exact_recovery_and_decline_in_both_units(self):
        # The old log1p(100) exceeds the 1.5 clip. With a reference of 100,
        # zero -> 100 needs only log(2); 100 -> 0 needs -log(2).
        last=np.array([0.,100.,100.,.2])
        actual=np.array([100.,0.,100.,.3])
        reference=np.array([100.,100.,100.,.2])
        expected=[np.log(2),-np.log(2),0.,np.log(1.25)]
        for factor in (.001,1.,1000.):
            delta=log_residual_target(actual*factor,last*factor,reference*factor)
            np.testing.assert_allclose(delta,expected,rtol=1e-12,atol=1e-14)
            predicted=log_residual_prediction(last*factor,reference*factor,delta)
            np.testing.assert_allclose(predicted/factor,actual,rtol=1e-12,atol=1e-12)

    def test_clip_is_dimensionless_and_never_observed_stays_zero(self):
        for factor in (.001,1.,1000.):
            pred=log_residual_prediction(np.array([0.,100.,100.,0.])*factor,
                     np.array([100.,100.,100.,0.])*factor,np.array([100.,100.,-100.,100.]))
            np.testing.assert_allclose(pred/factor,[348.1689070338065,796.337814067613,0.,0.],
                                       rtol=1e-12,atol=1e-12)
            self.assertTrue(np.isfinite(pred).all())

    def panel(self, factor=1., frequency='monthly', metric='volume_kg'):
        meta=np.array([['Tehran','Co','A','Sub','C','Not specified','Not specified','Not specified'],
                       ['Tehran','Co','B','Sub','C','Not specified','Not specified','Not specified'],
                       ['Shiraz','Co','Z','Sub','C','Not specified','Not specified','Not specified']])
        volume=np.array([[100.]*12+[0.,100.,100.,0.,100.,0.,200.,300.],
                         list(np.arange(20,dtype=float)+20),[0.]*20])
        value=volume*(2+np.arange(20)/10)
        if metric=='volume_kg':volume*=factor
        else:value*=factor
        dates=[str(p.asfreq('M','end')) for p in pd.period_range('2023-01',periods=20,
                            freq='2M' if frequency=='bimonthly' else 'M')]
        return Dataset(meta,volume,value,dates,'synthetic',periodicity=frequency)

    def test_training_and_inference_use_origin_reference_for_both_metrics(self):
        # Spy verifies the exact target entering fit; its controlled log(2)
        # prediction tests the inverse through fit_models, not just a helper.
        fitted=[]
        class Estimator:
            def __init__(self,**kwargs):pass
            def fit(self,x,y,**kwargs):
                fitted.append((x.copy(),np.asarray(y).copy()))
                return self
            def predict(self,x):return np.full(len(x),np.log(2))
            def save_model(self,path):Path(path).write_text('test')
        for metric in ('volume_kg','value_1000_toman'):
            for frequency in ('monthly','bimonthly'):
                for factor in (1.,1000.):
                    with self.subTest(metric=metric,frequency=frequency,factor=factor), \
                         tempfile.TemporaryDirectory() as folder, \
                         patch.object(auto_engine,'OUT',Path(folder)), \
                         patch.object(auto_engine,'CatBoostRegressor',Estimator):
                        fitted.clear()
                        d=self.panel(factor,frequency,metric)
                        y=d.metric(metric)
                        pred=auto_engine.fit_models(d,metric,17,'all')['log']
                        x,target=fitted[2]
                        # Origin 12: product A has latest 0, last positive y[0,11].
                        # H1 target is y[0,13]; scale cannot use that future target.
                        row=(x['Brand']=='A')&(x['origin_index']==12)&(x['horizon']==1)
                        self.assertEqual(int(row.sum()),1)
                        self.assertAlmostEqual(target[row.to_numpy()][0],
                            np.log1p(y[0,13]/y[0,11]),places=12)
                        # Origin 17: latest zero, last positive y[0,16].
                        np.testing.assert_allclose(pred[0],y[0,16],rtol=1e-12)
                        np.testing.assert_array_equal(pred[2],0)
                        # Changing unknown future sales leaves all output intact.
                        d.volume[:,18:]=1e8;d.value[:,18:]=1e9
                        fitted.clear()
                        with tempfile.TemporaryDirectory() as fresh, \
                             patch.object(auto_engine,'OUT',Path(fresh)):
                            again=auto_engine.fit_models(d,metric,17,'all')['log']
                        np.testing.assert_array_equal(pred,again)

if __name__=='__main__':
    unittest.main()
