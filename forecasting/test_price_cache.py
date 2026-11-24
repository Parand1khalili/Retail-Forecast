"""Price target/weight and actual runtime policy cache regression checks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import auto_engine as engine
from app_core import Dataset
from category_selection import CategoryGuard


def panel():
    meta=np.array([['Tehran','Co',b,'Sub','C','Not specified','Not specified','Not specified']
                   for b in ('A','B','C','Z')])
    q=np.tile(np.array([.2,.8,.4,0.])[:,None],(1,36))
    q[2,17]=0
    price=np.tile(np.arange(1,37,dtype=float)*1e-10,(4,1))
    dates=[str(x) for x in pd.period_range('2023-01',periods=36,freq='M')]
    return Dataset(meta,q,q*price,dates,'synthetic')


class PriceCacheTests(unittest.TestCase):
    def test_price_fit_uses_real_positive_ratio_and_relative_weight(self):
        fits=[]
        class Spy:
            def __init__(self,**kwargs):pass
            def fit(self,x,y,**kwargs):
                fits.append((x.copy(),np.array(y),kwargs['sample_weight']))
                return self
            def predict(self,x):return np.full(len(x),np.log(1.5))
            def save_model(self,path):Path(path).write_text('spy')
        for factor in (1.,1000.,.001):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory() as folder, \
                 patch.object(engine,'OUT',Path(folder)), patch.object(engine,'CatBoostRegressor',Spy):
                d=panel();d.volume*=factor
                fits.clear()
                p=engine.fit_models(d,'volume_kg',17,'all')['price']
                x,y,w=fits[-1]
                mask=(x.origin_index==11)&(x.horizon==1)
                rows=x.loc[mask,'Brand'].tolist()
                self.assertEqual(rows,['A','B','C'])
                np.testing.assert_allclose(y[mask],np.log(13/12),atol=1e-12)
                np.testing.assert_allclose(w[mask],np.sqrt(np.array([.2,.8,.4])*factor))
                # A/B have a valid current price; C is missing at the origin
                # and carries period 16's price; Z was never observed.
                expected=np.array([18e-10*1.5,18e-10*1.5,17e-10,0])/factor
                np.testing.assert_allclose(p,np.repeat(expected[:,None],3,axis=1),
                                           rtol=1e-12,atol=0)
                self.assertFalse(((x.Brand=='C')&(x.origin_index==17)).any())

    def test_policy_cache_tracks_model_and_actual_category_guard(self):
        d=panel()
        def candidate(d,metric,o):
            return {'base|Last month|2':np.repeat(d.metric(metric)[:,o,None],3,axis=1)}
        with tempfile.TemporaryDirectory() as folder, patch.object(engine,'OUT',Path(folder)), \
             patch.object(engine,'candidates',side_effect=candidate) as generated:
            engine.dataset_policy(d,'volume_kg',35)
            first=generated.call_count
            self.assertGreater(first,0)
            engine.dataset_policy(d,'volume_kg',35)
            self.assertEqual(generated.call_count,first)
            with patch.object(engine,'PARAMS',dict(engine.PARAMS,depth=engine.PARAMS['depth']+1)):
                engine.dataset_policy(d,'volume_kg',35)
            self.assertGreater(generated.call_count,first)
            count=generated.call_count
            with patch.object(engine,'CategoryGuard',return_value=CategoryGuard(min_series=99)):
                result=engine.dataset_policy(d,'volume_kg',35)
                self.assertEqual(result['_hierarchy']['guard']['min_series'],99)
                self.assertEqual(result['_validation']['category_guard']['min_series'],99)
            self.assertGreater(generated.call_count,count)
            self.assertEqual(len(list((Path(folder)/'runtime_policies').glob('*.json'))),3)


if __name__=='__main__':
    unittest.main()
