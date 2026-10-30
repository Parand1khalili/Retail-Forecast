"""Synthetic regression tests for shared CatBoost/light selection; no retail data needed."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_core import Dataset
import auto_engine as engine


class SharedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.months=[str(p) for p in pd.period_range('2024-01',periods=27,freq='M')]
        meta=np.array([
            ['Tehran','Co','A','Main','Yoghurt','Plain','X','X'],
            ['Tehran','Co','B','Main','Yoghurt','Plain','X','X'],
            ['Shiraz','Co','C','Main','Yoghurt','Plain','X','X']])
        volume=np.repeat(np.array([[100.],[60.],[0.]]),27,axis=1)
        self.dataset=Dataset(meta,volume,volume*10,self.months,'synthetic')

    @staticmethod
    def bad_ml(d,metric,cutoff,window,predict_only=None):
        wrong=np.broadcast_to(d.metric(metric)[:,cutoff,None]+5000,(len(d.meta),3)).copy()
        return {'raw':wrong,'scaled':wrong,'log':wrong,'price':np.ones((len(d.meta),3))}

    def test_shared_origins_and_bad_ml_rejected(self):
        d=self.dataset
        with patch.object(engine,'fit_models',self.bad_ml):
            for metric in ['volume_kg','value_1000_toman']:
                origins=[20,23]
                candidates={o:engine.candidates(d,metric,o) for o in origins}
                self.assertTrue(any(k.startswith('light|') for k in candidates[20]))
                self.assertTrue(any(k.startswith('ml|') for k in candidates[20]))
                policy,rows=engine.select_recipes(d,metric,origins,candidates)
                self.assertEqual({r['horizon'] for r in rows},{1,2,3})
                for h in [1,2,3]:
                    selected=policy[str(h)]
                    self.assertEqual(selected['validation_wape'],0)
                    self.assertEqual(selected['validation_origins'],[self.months[o] for o in origins])
                    self.assertEqual(selected['comparison_grain'],'city_segment')
                    self.assertFalse(selected['name'].startswith(('ml|','blend|','price-volume|')))
                    for name,_ in selected['components']:
                        self.assertEqual(engine.forecast_component(d,metric,23,name).shape,(3,3))

    def test_future_observations_do_not_change_predictions(self):
        d=self.dataset
        changed=Dataset(d.meta,d.volume.copy(),d.value.copy(),d.months,d.source)
        changed.volume[:,24:]=1234567
        changed.value[:,24:]=9876543
        with patch.object(engine,'fit_models',self.bad_ml):
            for metric in ['volume_kg','value_1000_toman']:
                original=engine.candidates(d,metric,23)
                altered=engine.candidates(changed,metric,23)
                self.assertEqual(original.keys(),altered.keys())
                for key in original:
                    np.testing.assert_array_equal(original[key],altered[key])

    def test_unobserved_validation_target_rejected(self):
        with patch.object(engine,'fit_models',self.bad_ml):
            observations={25:engine.candidates(self.dataset,'volume_kg',25)}
            with self.assertRaises(ValueError):
                engine.select_recipes(self.dataset,'volume_kg',[25],observations)


if __name__=='__main__':
    unittest.main()
