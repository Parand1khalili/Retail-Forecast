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
                    evidence=selected['selection_evidence']
                    self.assertEqual(evidence['selected'],selected['name'])
                    self.assertEqual(evidence['scope'],'all_series')
                    self.assertEqual(evidence['horizon'],h)
                    self.assertEqual(evidence['grain'],'city_segment')
                    self.assertEqual(len(evidence['origin_details']),len(origins))
                    self.assertEqual({x['target'] for x in evidence['origin_details']},
                                     {d.months[o+h] for o in origins})
                    self.assertTrue(all(x['wape']==0 for x in evidence['origin_details']))
                    self.assertTrue(all(x['absolute_error']==0 for x in evidence['origin_details']))
                    self.assertTrue(any(x['name'].startswith('ml|')
                                        for x in evidence['candidate_scores']))
                    self.assertTrue(any(len(x['components'])==2
                                        for x in evidence['candidate_scores']))
                    self.assertEqual(evidence['candidate_scores'][0]['wape'],0)
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

    def test_selection_diagnostics_match_global_policy_for_filtered_requests(self):
        import tempfile
        from pathlib import Path
        d=self.dataset
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(engine,'OUT',Path(cache)), patch.object(engine,'fit_models',self.bad_ml):
                all_series=engine.future_request(d,{'metric':'volume_kg','horizon':1,'filters':{}})
                brand_a=engine.future_request(d,{'metric':'volume_kg','horizon':1,
                                                 'filters':{'Brand':'A'}})
                brand_b=engine.future_request(d,{'metric':'volume_kg','horizon':1,
                                                 'filters':{'Brand':'B'}})
        for result in [all_series,brand_a,brand_b]:
            evidence=result['selection_evidence']
            self.assertEqual(evidence['selected'],result['calibration']['name'])
            self.assertEqual(evidence['scope'],'all_series')
            self.assertAlmostEqual(evidence['candidate_scores'][0]['wape'],0)
            self.assertEqual(len(evidence['origin_details']),2)
            self.assertEqual(evidence['horizon'],1)
        self.assertEqual(all_series['selection_evidence'],brand_a['selection_evidence'])
        self.assertEqual(brand_a['selection_evidence'],brand_b['selection_evidence'])
        self.assertNotEqual(brand_a['prediction'],brand_b['prediction'])

    def test_short_history_has_no_fake_comparison(self):
        d=self.dataset
        short=Dataset(d.meta,d.volume[:,:12],d.value[:,:12],d.months[:12],d.source)
        result=engine.future_request(short,{'metric':'volume_kg','horizon':1})
        self.assertIsNone(result['selection_evidence'])
        self.assertIsNone(result['calibration']['validation_wape'])

    def test_unobserved_validation_target_rejected(self):
        with patch.object(engine,'fit_models',self.bad_ml):
            observations={25:engine.candidates(self.dataset,'volume_kg',25)}
            with self.assertRaises(ValueError):
                engine.select_recipes(self.dataset,'volume_kg',[25],observations)


if __name__=='__main__':
    unittest.main()
