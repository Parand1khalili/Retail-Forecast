"""Regression tests for unit-covariant scale and true parent ratios."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import auto_engine
from app_core import Dataset, DIMS
from auto_engine import Features, fit_models


class FeatureMathTests(unittest.TestCase):
    def panel(self, frequency='monthly'):
        # Two subbrands of A share the same parent; B is a different brand.
        # Another city has no sales. D has been dormant for >12 observations.
        meta=np.array([
            ['Tehran','Co','A','A1','C','Not specified','Not specified','Not specified'],
            ['Tehran','Co','A','A2','C','Not specified','Not specified','Not specified'],
            ['Tehran','Co','B','B1','C','Not specified','Not specified','Not specified'],
            ['Shiraz','Co','Z','Z1','C','Not specified','Not specified','Not specified'],
            ['Rasht','Co','D','D1','C','Not specified','Not specified','Not specified'],
            ['Karaj','Co','N','N1','C','Not specified','Not specified','Not specified'],
        ])
        y=np.zeros((6,24))
        y[0,12:18]=[.1,.1,.1,.2,.2,.2]
        y[1,12:18]=.3
        y[2,12:18]=.5
        y[4,1]=.8
        y[5,15:18]=.25
        y[:3,18:]=[[.4,.1,.3,.6,.2,.1],
                    [.1,.2,.3,.4,.1,.2],
                    [.2,.3,.4,.5,.6,.7]]
        y[4,19]=.4
        # Nonconstant positive unit values permit real price-model training.
        money=y*(2+np.arange(24)/20+np.arange(6)[:,None]/10)
        freq='2M' if frequency=='bimonthly' else 'M'
        months=[str(p.asfreq('M','end')) for p in pd.period_range('2023-01',periods=24,freq=freq)]
        return Dataset(meta,y,money,months,'synthetic',periodicity=frequency)

    def test_parent_ratios_use_actual_denominator_and_group_members(self):
        d=self.panel()
        x,_,_=Features(d,'volume_kg',0,17).features(17,1)
        np.testing.assert_allclose(x['share_of_brand_city'][:3],[.4,.6,1])
        np.testing.assert_allclose(x['share_of_market_city'][:3],[.2,.3,.5])
        np.testing.assert_allclose(x['brand_city_growth3'][:3],[.25,.25,0],atol=1e-14)
        np.testing.assert_allclose(x['market_city_growth3'][:3],[1/9]*3)
        # A real zero numerator is a zero share when the parent is positive.
        # Zero parent totals and growth from zero are undefined.
        self.assertTrue(np.isnan(x.loc[3,'share_of_brand_city']))
        self.assertTrue(np.isnan(x.loc[3,'market_city_growth3']))
        self.assertTrue(np.isnan(x.loc[4,'share_of_brand_city']))
        self.assertTrue(np.isnan(x.loc[5,'brand_city_growth3']))
        self.assertEqual(x.loc[5,'share_of_market_city'],1)

    def test_scale_and_normalized_features_are_unit_invariant(self):
        for frequency in ('monthly','bimonthly'):
            for metric in ('volume_kg','value_1000_toman'):
                with self.subTest(frequency=frequency,metric=metric):
                    d=self.panel(frequency)
                    x,s,ever=Features(d,metric,0,17).features(17,3)
                    actual=d.metric(metric)[:,18]
                    latest=d.metric(metric)[:,17]
                    for factor in (.001,1000.):
                        other=self.panel(frequency)
                        if metric=='volume_kg':other.volume*=factor
                        else:other.value*=factor
                        z,ss,active=Features(other,metric,0,17).features(17,3)
                        np.testing.assert_array_equal(ever,active)
                        np.testing.assert_allclose(ss,s*factor,rtol=1e-12,atol=0)
                        columns=['recent_growth','last_change','share_of_brand_city',
                                 'share_of_market_city','brand_city_growth3','market_city_growth3']
                        np.testing.assert_allclose(z[columns],x[columns],rtol=1e-12,atol=1e-14)
                        expected=(actual[ever]-latest[ever])/s[ever]
                        target=(other.metric(metric)[ever,18]-other.metric(metric)[ever,17])/ss[ever]
                        np.testing.assert_allclose(target,expected,rtol=1e-12,atol=1e-14)
                        # Existing sqrt(scale) weighting changes by a common
                        # factor only, preserving relative sample weights.
                        np.testing.assert_allclose(np.sqrt(ss[ever]),np.sqrt(s[ever])*np.sqrt(factor))
                        np.testing.assert_allclose(latest[ever]+s[ever]*expected,actual[ever],atol=1e-14)

    def test_zero_and_dormant_scales_are_safe_and_prefix_only(self):
        d=self.panel()
        x,s,ever=Features(d,'volume_kg',0,17).features(17,1)
        np.testing.assert_allclose(s,[.2,.3,.5,0,.8,.25])
        np.testing.assert_array_equal(ever,[True,True,True,False,True,True])
        np.testing.assert_allclose(x['recent_growth'],[.5,0,0,0,0,1],atol=1e-14)
        np.testing.assert_allclose(x['last_change'],0,atol=1e-14)
        self.assertTrue((s[ever]>0).all())
        # A future launch must not give an unobserved series a scale.
        d.volume[:,18:]=999
        d.value[:,18:]=9999
        z,ss,active=Features(d,'volume_kg',0,23).features(17,1)
        np.testing.assert_array_equal(ss,s)
        np.testing.assert_array_equal(active,ever)
        pd.testing.assert_frame_equal(z,x)

    def test_zero_share_and_complete_decline(self):
        d=self.panel()
        d.volume[0,15:18]=0
        x,_,_=Features(d,'volume_kg',0,17).features(17,1)
        self.assertEqual(x.loc[0,'share_of_brand_city'],0)
        self.assertEqual(x.loc[1,'share_of_brand_city'],1)
        d.volume[:2,15:18]=0
        x,_,_=Features(d,'volume_kg',0,17).features(17,1)
        np.testing.assert_allclose(x['brand_city_growth3'][:2],[-1,-1])
        self.assertTrue(np.isnan(x.loc[0,'share_of_brand_city']))
        self.assertEqual(x.loc[0,'share_of_market_city'],0)

    def test_real_catboost_fits_with_small_and_dormant_series(self):
        d=self.panel()
        params=dict(auto_engine.PARAMS,iterations=8,thread_count=1)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(auto_engine,'OUT',Path(folder)), \
             patch.object(auto_engine,'PARAMS',params):
            for metric in ('volume_kg','value_1000_toman'):
                with self.subTest(metric=metric):
                    result=fit_models(d,metric,17,'all')
                    self.assertIn('scaled',result)
                    self.assertIn('raw',result)
                    self.assertIn('log',result)
                    for prediction in result.values():
                        self.assertEqual(prediction.shape,(6,3))
                        self.assertTrue(np.isfinite(prediction).all())
                        self.assertTrue((prediction>=0).all())
                        np.testing.assert_array_equal(prediction[3],0)
                    cached=fit_models(d,metric,17,'all')
                    for name in result:
                        np.testing.assert_array_equal(cached[name],result[name])


if __name__=='__main__':
    unittest.main()
