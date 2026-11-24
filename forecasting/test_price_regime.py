"""Numerically independent counterexamples for the conditional Value policy."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_core import Dataset, DIMS
import auto_engine as engine
import evaluate_hierarchy
from full_export import _modelled_rows
from price_regime import GUARD_ID, shock_origins, conditional_policy
from selection_core import candidate_info


def panel(periods=48, frequency='monthly'):
    step=2 if frequency=='bimonthly' else 1
    months=[str(pd.Period('2020-01',freq='M')+i*step) for i in range(periods)]
    meta=np.array([['Tehran','Co',f'{category}{i}','Sub',category,
                   'Not specified','Not specified','Not specified']
                   for category in ('A','B') for i in range(8)])
    volume=np.full((16,periods),20.)
    return Dataset(meta,volume,volume*10,months,'synthetic-price-regime',
                   dimension_names=DIMS,periodicity=frequency)


def normal_policy():
    return {**{str(h):{'name':'base|A','components':[['base|A',1.]],
                        'selection_evidence':{'candidate_scores':[]}}
               for h in (1,2,3)},
            '_hierarchy':{'dimensions':['Subcategory'],'overrides':{},
                          'groups_considered':0,'groups_selected':0}}


class ConditionalMathTests(unittest.TestCase):
    def test_registry_only_value_and_history(self):
        spec=candidate_info(GUARD_ID)
        self.assertEqual(spec.metrics,('value_1000_toman',))
        self.assertEqual(spec.min_history,18)
        self.assertEqual(set(spec.periodicities),{'monthly','bimonthly'})

    def test_event_origins_ignore_opposite_direction_overlap_and_current_episode(self):
        # Consecutive detections 17/18 are one episode. The event at 23
        # is too close to the event at 25; the 34/35/36 episode is current.
        signs={17:1,18:1,21:-1,23:1,25:1,30:1,34:1,35:1,36:1}
        def detect(o):
            s=signs.get(o,0)
            return {'active':bool(s),'current_change':s*.4,'median_prior_change':0.}
        self.assertEqual(shock_origins(36,detect),[17,25,30])
        self.assertEqual(shock_origins(33,detect),[])

    def test_insufficient_events_keeps_normal_policy_including_category(self):
        normal=normal_policy()
        normal['_hierarchy']['overrides']['1']=[{
            'name':'base|B','components':[['base|B',1.]],'path':['B'],'depth':1}]
        p=conditional_policy(panel(),47,normal,[17],{},set())
        self.assertEqual(p['1'],normal['1'])
        self.assertEqual(p['_hierarchy'],normal['_hierarchy'])
        self.assertEqual(p['_price_regime']['reason'],'insufficient_events')

    def test_per_horizon_and_category_compete_with_independent_expected_winners(self):
        d=panel()
        origins=[17,24,31]
        grid={}
        # Actual 200 in each cell. Reference error 20. Specialist wins H1
        # on A, but B is better with base B. Specialist loses H2/H3.
        for o in origins:
            a=np.full((16,3),220.)
            b=np.full((16,3),245.);b[8:,:]=200.
            special=np.full((16,3),240.);special[:8,0]=200.
            grid[o]={'base|A':a,'base|B':b,GUARD_ID:special}
        p=conditional_policy(d,47,normal_policy(),origins,grid,set(grid[17]))
        self.assertEqual(p['1']['name'],'base|A')  # tie in pooled error with guard
        self.assertEqual(p['2']['name'],'base|A')
        self.assertEqual(p['3']['name'],'base|A')
        from category_selection import series_recipes
        for h in (1,2,3):
            names=[r['name'] for r in series_recipes(d,p[str(h)],p['_hierarchy'],h)]
            self.assertEqual(names[:8],[GUARD_ID if h==1 else 'base|A']*8)
            self.assertEqual(names[8:],['base|B']*8)
        self.assertEqual(p['_price_regime']['horizons']['1']['candidate_scores'][0]['absolute_error'],960.)

    def test_unavailable_specialist_and_nonfinite_predictions_do_not_compete(self):
        d=panel();origins=[17,24]
        grid={o:{'base|A':np.full((16,3),220.),GUARD_ID:np.full((16,3),200.)}
              for o in origins}
        grid[24][GUARD_ID][0,0]=np.nan
        p=conditional_policy(d,47,normal_policy(),origins,grid,set(grid[17]))
        self.assertEqual(p['_price_regime']['reason'],'guard_unavailable')
        self.assertEqual(p['1']['name'],'base|A')


class ProductionRegimeTests(unittest.TestCase):
    @staticmethod
    def ordinary_candidates(d,metric,origin):
        # Keep the actual statistical implementation; restrict the candidate
        # set to a two-period mean for transparent hand-derived expectations.
        return {'base|Mean 2|2':engine.base_candidates(d,metric,origin)['base|Mean 2|2']}

    @staticmethod
    def volume_fit(d,metric,cutoff,window,predict_only=None):
        return {'raw':np.repeat(d.volume[:,cutoff,None],3,axis=1)}

    def test_real_forecast_no_forced_override_and_report_replays_same_cutoff(self):
        d=panel(36)
        # Origin 32: only observed spike, 400 value, then return to 200.
        d.value[:,32]=400.
        with TemporaryDirectory() as temp, \
             patch.object(engine,'OUT',Path(temp)), \
             patch.object(engine,'candidates',side_effect=self.ordinary_candidates), \
             patch.object(evaluate_hierarchy,'candidates',side_effect=self.ordinary_candidates), \
             patch.object(engine,'fit_models',side_effect=self.volume_fit):
            for h in (1,2,3):
                live, recipe=engine.automatic_forecast(d,'value_1000_toman',h,cutoff=32)
                np.testing.assert_array_equal(live,np.full(16,300.))
                self.assertEqual(recipe['price_regime']['reason'],'insufficient_events')
                self.assertEqual(recipe['price_regime']['specialist_series'],0)
            rows=evaluate_hierarchy.report(d,'value_1000_toman',Path(temp)/'summary.csv')
            for row in rows:
                self.assertEqual(row['live_audit_scope'],'cutoff_safe_policy_replay')
                self.assertEqual(row['previous_live_wape'],1.)
                self.assertEqual(row['live_hierarchical_wape'],.5)
                self.assertEqual(row['live_relative_reduction_vs_previous'],.5)

    def test_real_candidate_wins_only_with_repeatable_evidence_and_export_matches(self):
        for frequency in ('monthly','bimonthly'):
            with self.subTest(frequency=frequency):
                d=panel()
                for origin in (17,24,32,47):
                    d.value[:,origin:]*=1.5
                with TemporaryDirectory() as temp, \
                     patch.object(engine,'OUT',Path(temp)), \
                     patch.object(engine,'candidates',side_effect=self.ordinary_candidates), \
                     patch.object(engine,'fit_models',side_effect=self.volume_fit):
                    for h in (1,2,3):
                        result, recipe=engine.automatic_forecast(d,'value_1000_toman',h)
                        # Known quantity is constant and prior median price growth
                        # zero, so all forecasts equal the latest observed value.
                        np.testing.assert_allclose(result,d.value[:,-1])
                        self.assertEqual(recipe['name'],GUARD_ID)
                        self.assertEqual(recipe['price_regime']['specialist_series'],16)
                        self.assertEqual(len(recipe['price_regime']['origins']),3)
                    interactive=engine.future_request(d,{'metric':'value_1000_toman','horizon':1,
                                                           'filters':{'Subcategory':'A'}})
                    output=_modelled_rows(d)
                    self.assertAlmostEqual(interactive['prediction'],
                                           output['value_1000_toman'][:8,0].sum())

    def test_actual_catboost_specialist_is_finite_and_cutoff_safe(self):
        # Real CatBoost fit (small deterministic configuration for CI);
        # constant quantity has zero residual, so the exact expected result
        # at a persistent 50% price jump is 300 for all horizons.
        d=panel(28)
        d.value[:,23:]*=1.5
        params={**engine.PARAMS,'iterations':4,'depth':2,'thread_count':1}
        with TemporaryDirectory() as temp, \
             patch.object(engine,'OUT',Path(temp)), \
             patch.object(engine,'PARAMS',params):
            pred=engine.forecast_component(d,'value_1000_toman',23,GUARD_ID)
            np.testing.assert_allclose(pred,np.full((16,3),300.),rtol=1e-8)
            self.assertTrue(list(Path(temp).rglob('raw.cbm')))
            d.volume[:,24:]=9000.;d.value[:,24:]=900000.
            again=engine.forecast_component(d,'value_1000_toman',23,GUARD_ID)
            np.testing.assert_array_equal(pred,again)
            with self.assertRaisesRegex(ValueError,'unavailable'):
                engine.forecast_component(d,'value_1000_toman',22,GUARD_ID)

    def test_real_nonconstant_catboost_specialist_composes_volume_and_unit_value(self):
        d=panel(28)
        d.volume[:,:]=20.+np.arange(28)[None,:]*.3+np.arange(16)[:,None]*.1
        d.value[:,:]=d.volume*10.
        d.value[:,23:]*=1.5
        params={**engine.PARAMS,'iterations':4,'depth':2,'thread_count':1}
        with TemporaryDirectory() as temp, \
             patch.object(engine,'OUT',Path(temp)), \
             patch.object(engine,'PARAMS',params):
            raw=engine.fit_models(d,'volume_kg',23,'all')['raw']
            forecast=engine.forecast_component(d,'value_1000_toman',23,GUARD_ID)
            expected=15.*(.5*raw+.5*d.volume[:,23,None])
            np.testing.assert_allclose(forecast,expected,rtol=1e-8)
            self.assertTrue(np.isfinite(forecast).all())
            self.assertTrue(list(Path(temp).rglob('raw.cbm')))

    def test_future_perturbation_cannot_change_historical_policy_or_forecast(self):
        d=panel()
        for origin in (17,24,32,41):
            d.value[:,origin:]*=1.5
        with TemporaryDirectory() as temp, \
             patch.object(engine,'OUT',Path(temp)), \
             patch.object(engine,'candidates',side_effect=self.ordinary_candidates), \
             patch.object(engine,'fit_models',side_effect=self.volume_fit):
            before, recipe1=engine.automatic_forecast(d,'value_1000_toman',2,cutoff=41)
            d.volume[:,42:]=1e6;d.value[:,42:]=1e12
            after, recipe2=engine.automatic_forecast(d,'value_1000_toman',2,cutoff=41)
            np.testing.assert_array_equal(before,after)
            self.assertEqual(recipe1,recipe2)


if __name__=='__main__':
    unittest.main()
