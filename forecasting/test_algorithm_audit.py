"""Independent counterexamples for algorithmic correctness, without private data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_core import Dataset, DIMS
from auto_engine import data_identity, fit_models, dataset_policy
import auto_engine
from selection_core import choose_champion
from category_selection import select_category_overrides


class AlgorithmAuditTests(unittest.TestCase):
    def test_relative_guard_does_not_depend_on_grams_vs_kg(self):
        # Reference is worse overall, but the challenger is 20% worse on
        # one origin. It must be rejected regardless of the unit.
        for factor in (0.001, 1., 1000.):
            losses = {'reference': np.array([.5,.5,.1])*factor,
                      'challenger': np.array([.1,.1,.12])*factor}
            winner, _ = choose_champion(losses, 'reference')
            self.assertEqual(winner, 'reference')

    def test_perfect_origin_does_not_allow_positive_error(self):
        winner, _ = choose_champion({'reference':[10.,10.,0.],
                                      'challenger':[1.,1.,.01]}, 'reference')
        self.assertEqual(winner,'reference')

    def test_stable_category_runner_up_is_not_vetoed_by_unstable_best(self):
        # 8 independent series, 20 actual each. Per-cell errors over three
        # origins are parent=(10,10,10), unstable=(0,0,11), stable=(6,6,6).
        # Unstable has the smallest pooled error but fails the every-origin
        # guard. Stable is 40% better at each origin and should be selected.
        meta=np.array([['Tehran','Co',f'Brand-{i}','Sub','A',
                       'Not specified','Not specified','Not specified']
                       for i in range(8)])
        months=[str(x) for x in pd.period_range('2023-01',periods=36,freq='M')]
        actual=np.full((8,36),20.)
        d=Dataset(meta,actual,actual*10,months,'synthetic')
        origins=[17,20,23]
        predictions={}
        for j,o in enumerate(origins):
            predictions[o]={'base|A':np.full((8,3),30.),
                            'base|B':np.full((8,3),20.+[0,0,11][j]),
                            'base|C':np.full((8,3),26.)}
        catalog=[{'name':name,'components':[[name,1.]]}
                 for name in ('base|A','base|B','base|C')]
        parent={str(h):{'name':'base|A','components':[['base|A',1.]],
                        'selection_evidence':{'candidate_scores':catalog}}
                for h in (1,2,3)}
        result=select_category_overrides(d,'volume_kg',origins,predictions,parent)
        for h in ('1','2','3'):
            chosen=result['overrides'][h]
            self.assertEqual(len(chosen),1)
            self.assertEqual(chosen[0]['name'],'base|C')
            self.assertAlmostEqual(chosen[0]['evidence']['relative_improvement'],.4)

    def panel(self, start='2023-01', unit='kg', object_meta=False):
        meta=np.array([['Tehran','Co','A','Sub','Category',
                       'Not specified','Not specified','Not specified']],
                       dtype=object if object_meta else str)
        months=[str(x) for x in pd.period_range(start,periods=36,freq='M')]
        volume=np.full((1,36),20.)
        return Dataset(meta,volume,volume*10,months,'synthetic',
                       volume_unit=unit,dimension_names=DIMS)

    def test_cache_identity_includes_calendar_units_and_only_available_data(self):
        d=self.panel()
        first=data_identity(d,0,23)
        self.assertNotEqual(first,data_identity(self.panel('2023-02'),0,23))
        self.assertNotEqual(first,data_identity(self.panel(unit='gr'),0,23))
        self.assertEqual(first,data_identity(self.panel(object_meta=True),0,23))
        d.volume[:,24:]=999.
        d.value[:,24:]=99999.
        self.assertEqual(first,data_identity(d,0,23))

    def test_actual_policy_cache_does_not_return_stale_validation_dates(self):
        a=self.panel('2023-01')
        b=self.panel('2023-02')
        def simple(d,metric,cutoff):
            return {'base|Last month|2':np.repeat(d.metric(metric)[:,cutoff,None],3,axis=1)}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(auto_engine,'OUT',Path(folder)), \
             patch.object(auto_engine,'candidates',side_effect=simple):
            x=dataset_policy(a,'volume_kg',23,'global')
            y=dataset_policy(b,'volume_kg',23,'global')
            self.assertEqual(x['_selection_origins'],[a.months[17],a.months[20]])
            self.assertEqual(y['_selection_origins'],[b.months[17],b.months[20]])
            self.assertNotEqual(x['_selection_origins'],y['_selection_origins'])

    def test_real_fit_cache_recomputes_when_calendar_changes(self):
        # An independent fake estimator deliberately uses the calendar feature.
        # This checks the actual cache path, not merely a helper's return value.
        class CalendarEstimator:
            def __init__(self,**kwargs): pass
            def fit(self,x,y,**kwargs): return self
            def predict(self,x): return x['target_month'].to_numpy(float)
            def save_model(self,path): Path(path).write_text('test model')
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(auto_engine,'OUT',Path(folder)), \
             patch.object(auto_engine,'CatBoostRegressor',CalendarEstimator):
            a=fit_models(self.panel('2023-01'),'volume_kg',23,'all')
            b=fit_models(self.panel('2023-02'),'volume_kg',23,'all')
            self.assertFalse(np.array_equal(a['raw'],b['raw']))
            self.assertEqual(a['raw'][0,0],21.)
            self.assertEqual(b['raw'][0,0],22.)


if __name__=='__main__':
    unittest.main()
