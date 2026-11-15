"""Full production selector tests using synthetic candidate arrays, no CatBoost mock-fitting.

The expected winner is derived independently from per-origin absolute errors.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import auto_engine


class FakeDataset:
    def __init__(self):
        self.meta=np.asarray([['All','A'],['All','B']],dtype=object)
        self.volume=np.full((2,36),20.,dtype=float)
        self.value=np.full((2,36),200.,dtype=float)
        self.months=[str(x) for x in pd.period_range('2023-01',periods=36,freq='M')]
        self.periodicity='monthly'
        self.period_months=1

    def all_dimensions(self):
        return ['Market','Subcategory']

    def metric(self, metric):
        return self.volume if metric=='volume_kg' else self.value


def samples(origins, stable=False):
    collected={}
    for origin in origins:
        reference=np.array([[20.,20.,20.],[30.,30.,30.]])
        challenger=np.full((2,3),20.) if stable or origin==origins[0] else np.full((2,3),26.)
        collected[origin]={'base|A':reference,'ml|raw|all':challenger}
    return collected


class ProductionSelectorTests(unittest.TestCase):
    def setUp(self):
        self.data=FakeDataset()
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache=Path(folder.name)

    def test_guard_rejects_global_winner_from_one_noisy_origin(self):
        origins=[23,26,29]
        d=self.data
        values=samples(origins)
        guarded,_=auto_engine.select_recipes(d,'volume_kg',origins,values)
        legacy,_=auto_engine.select_recipes(d,'volume_kg',origins,values,
                                            selection_mode='legacy')
        self.assertEqual(legacy['1']['name'],'ml|raw|all')
        self.assertEqual(guarded['1']['name'],'base|A')
        self.assertEqual(guarded['1']['selection_decision']['reason'],
                         'no challenger passed stability guard')
        self.assertEqual(guarded['1']['selection_evidence']['selection_decision']['legacy_best'],
                         'ml|raw|all')

    def test_incompatible_candidate_shape_never_enters_comparison(self):
        origins=[23,26,29]
        predictions=samples(origins,stable=True)
        for data in predictions.values():
            data['ml|raw|all']=np.zeros((2,2))
        policy,_=auto_engine.select_recipes(self.data,'volume_kg',origins,predictions)
        self.assertEqual(policy['1']['name'],'base|A')
        self.assertEqual(policy['1']['selection_decision']['candidate_count'],1)

    def test_guard_accepts_consistent_global_improvement(self):
        origins=[23,26,29]
        guarded,_=auto_engine.select_recipes(self.data,'volume_kg',origins,
                                             samples(origins,stable=True))
        self.assertEqual([guarded[str(h)]['name'] for h in (1,2,3)],
                         ['ml|raw|all']*3)
        self.assertEqual(guarded['1']['selection_decision']['mode'],'guarded')

    def test_horizon_specific_champion(self):
        origins=[23,26,29]
        data=samples(origins,stable=True)
        for item in data.values():
            item['ml|raw|all'][:,1:]=40.
        policy,_=auto_engine.select_recipes(self.data,'volume_kg',origins,data)
        self.assertEqual(policy['1']['name'],'ml|raw|all')
        self.assertEqual(policy['2']['name'],'base|A')
        self.assertEqual(policy['3']['name'],'base|A')

    def test_rolling_policy_reports_holdout_and_preserves_legacy_same_folds(self):
        d=self.data
        with patch.object(auto_engine,'OUT',self.cache), \
             patch.object(auto_engine,'candidates',
                          side_effect=lambda d,metric,cutoff: samples([cutoff],stable=True)[cutoff]):
            p=auto_engine.dataset_policy(d,'volume_kg',35,'hierarchical')
        self.assertEqual(p['_selection_origins'],
                         [d.months[i] for i in (23,26,29)])
        self.assertEqual(p['_holdout']['origin'],d.months[32])
        self.assertFalse(p['_holdout']['used_for_selection'])
        self.assertIn('legacy_global',p['_holdout']['horizons']['1'])
        self.assertIn('Subcategory',p['_holdout']['horizons']['1']['level_summary'])
        self.assertEqual(p['_validation']['mode'],'guarded')
        self.assertEqual(p['_legacy_global']['1']['name'],'ml|raw|all')

    def test_policy_cache_key_does_not_use_future_periods(self):
        d=self.data
        origins=[]
        def fake_candidates(dataset,metric,cutoff):
            origins.append(cutoff)
            return samples([cutoff],stable=True)[cutoff]
        with patch.object(auto_engine,'OUT',self.cache), \
             patch.object(auto_engine,'candidates',side_effect=fake_candidates):
            first=auto_engine.dataset_policy(d,'volume_kg',35)
            d.volume[:,35:]=999.
            second=auto_engine.dataset_policy(d,'volume_kg',34)
        self.assertNotEqual(first['_selection_origins'],second['_selection_origins'])
        self.assertTrue(all(o<=35 for o in origins))

    def test_short_history_returns_global_without_fake_independent_holdout(self):
        d=self.data
        with patch.object(auto_engine,'OUT',self.cache), \
             patch.object(auto_engine,'candidates',
                          side_effect=lambda d,metric,cutoff:samples([cutoff],stable=True)[cutoff]):
            p=auto_engine.dataset_policy(d,'volume_kg',24)
        self.assertIsNone(p['_holdout'])
        self.assertEqual(p['_validation']['mode'],'legacy')


if __name__=='__main__':
    unittest.main()
