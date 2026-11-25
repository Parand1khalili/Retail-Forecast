"""Adaptive selection, weighted MAE and end-to-end recipe/export parity."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import auto_engine as engine
from app_core import Dataset, DIMS
import full_export
from category_selection import category_dimensions, series_recipes
import test_feature_math
import walk_forward_audit


def panel():
    rows=[]
    for brand in ('A','B','C'):
        for city in ('Tehran','Shiraz'):
            for i in range(6):
                rows.append([city,'Co',brand,f'Item{i}','Juice','Single','Nectar',
                             'Not specified' if brand=='A' else 'Type','City'])
    meta=np.array(rows)
    data=np.full((len(rows),36),20.)
    return Dataset(meta,data.copy(),data.copy(),
                   [str(p) for p in pd.period_range('2023-01',periods=36,freq='M')],
                   'synthetic',dimension_names=DIMS+['Market'])


def candidates(d,metric,o,*,adaptive=False):
    a=np.full((len(d.meta),3),20.)
    b=np.full_like(a,40.)
    brand=d.meta[:,2]
    city=d.meta[:,0]
    local=(brand=='A')&(city=='Tehran')
    a[local,0]=30.
    b[local,0]=31. if o==29 else 20.
    a[brand=='B',0]=25.
    b[brand=='B',0]=20.
    # Only H1 benefits from the alternative.
    return {'base|A':a,'base|B':b}


class AdaptiveTests(unittest.TestCase):
    def test_local_brand_city_and_missing_category_fallback_in_live_pipeline(self):
        d=panel()
        def component(d,metric,cutoff,name):
            return candidates(d,metric,cutoff)[name]
        with tempfile.TemporaryDirectory() as folder, patch.object(engine,'OUT',Path(folder)), \
             patch.object(engine,'candidates',side_effect=candidates), \
             patch.object(engine,'forecast_component',side_effect=component):
            stable=engine.dataset_policy(d,'volume_kg',35,'hierarchical')
            adaptive=engine.dataset_policy(d,'volume_kg',35,'adaptive')
            self.assertEqual(stable['1']['name'],'base|A')
            self.assertEqual(adaptive['1']['name'],'base|A')
            self.assertEqual(adaptive['1']['selection_decision']['mode'],'minimum_wape')
            self.assertEqual(category_dimensions(d,adaptive=True),
                ['Subcategory','Subcategory2','Subcategory3','Subcategory4','Company','Brand','City'])
            recipes=series_recipes(d,adaptive['1'],adaptive['_hierarchy'],1)
            self.assertEqual(recipes[0]['path'],['Juice','Single','Nectar','Not specified','Co','A','Tehran'])
            self.assertEqual(recipes[0]['name'],'base|B')
            self.assertEqual(recipes[6]['name'],'base|A')
            self.assertEqual(recipes[12]['name'],'base|B')
            self.assertEqual(recipes[24]['name'],'base|A')
            self.assertTrue(all(r['name']=='base|A' for r in
                series_recipes(d,adaptive['2'],adaptive['_hierarchy'],2)))
            result=engine.future_request(d,{'metric':'volume_kg','horizon':1,
                'filters':{'Market':'City','City':'Tehran','Brand':'A'},'strategy':'adaptive'})
            self.assertEqual(result['prediction'],120.)
            self.assertEqual(result['category_selection']['selected_groups'][0]['recipe'],'base|B')
            predictions=full_export._modelled_rows(d,'adaptive')
            for metric in ('volume_kg','value_1000_toman'):
                np.testing.assert_array_equal(predictions[metric],np.full((36,3),20.))
            # Original hierarchy has no brand/city-specific recipe.
            original=full_export._modelled_rows(d,'hierarchical')
            self.assertEqual(original['volume_kg'][0,0],30.)
            self.assertEqual(original['volume_kg'][12,0],25.)

    def test_global_minimum_wape_does_not_require_two_percent_gain(self):
        d=panel();origins=[23,26,29]
        grid={o:{'base|A':np.full((36,3),30.),
                 'ml|raw|all':np.full((36,3),29.9)} for o in origins}
        stable,_=engine.select_recipes(d,'volume_kg',origins,grid)
        adaptive,_=engine.select_recipes(d,'volume_kg',origins,grid,selection_mode='minimum_wape')
        self.assertEqual(stable['1']['name'],'base|A')
        self.assertEqual(adaptive['1']['name'],'ml|raw|all')

    def test_actual_mae_fit_and_cache_are_isolated_from_existing_models(self):
        d=test_feature_math.FeatureMathTests().panel()
        with tempfile.TemporaryDirectory() as folder, patch.object(engine,'OUT',Path(folder)), \
             patch.object(engine,'PARAMS',dict(engine.PARAMS,iterations=8,thread_count=1)):
            for metric in ('volume_kg','value_1000_toman'):
                base=engine.fit_models(d,metric,17,'all')
                mae=engine.fit_models(d,metric,17,'all',mae_only=True)
                self.assertNotIn('scaled_mae',base)
                self.assertEqual(set(mae),{'scaled_mae'})
                p=mae['scaled_mae']
                self.assertEqual(p.shape,(6,3))
                self.assertTrue(np.isfinite(p).all() and (p>=0).all())
                np.testing.assert_array_equal(p[3],0.)
                np.testing.assert_array_equal(engine.forecast_component(d,metric,17,'ml|scaled_mae|all'),p)
                for key,value in base.items():
                    np.testing.assert_array_equal(engine.fit_models(d,metric,17,'all')[key],value)

    def test_mae_target_weight_and_inverse_match_native_absolute_error(self):
        fits=[]
        class Spy:
            def __init__(self,**kwargs):self.options=kwargs
            def fit(self,x,y,**kwargs):
                fits.append((self.options,x.copy(),np.asarray(y),kwargs['sample_weight']))
                return self
            def predict(self,x):return np.full(len(x),.25)
            def save_model(self,path):Path(path).write_text('test')
        d=test_feature_math.FeatureMathTests().panel()
        with tempfile.TemporaryDirectory() as folder, patch.object(engine,'OUT',Path(folder)), \
             patch.object(engine,'CatBoostRegressor',Spy):
            result=engine.fit_models(d,'volume_kg',17,'all',mae_only=True)['scaled_mae']
        options,x,target,weight=fits[0]
        self.assertEqual(options['loss_function'],'MAE')
        # Origin 14 -> 15: A1 grows .1 to .2; A2 remains .3.
        # Three observed recent values are .1 and .3 respectively.
        ix=(x['origin_index']==14)&(x['horizon']==1)&(x['Brand']=='A')
        self.assertEqual(int(ix.sum()),2)
        scales=np.array([.1,.3])
        np.testing.assert_allclose(target[ix],[1.,0.],atol=1e-12)
        np.testing.assert_allclose(weight[ix]/weight[ix][1],scales/scales[1],rtol=1e-12)
        # With a controlled normalized prediction .25, native absolute
        # errors are .075 and .075, despite different normalized errors.
        native=scales*np.abs(target[ix]-.25)
        np.testing.assert_allclose(native,[.075,.075],rtol=1e-12)
        normalized=weight[ix]*np.abs(target[ix]-.25)
        self.assertAlmostEqual(normalized[0],normalized[1],places=12)
        np.testing.assert_allclose(result[:2],[[.25]*3,[.375]*3],rtol=1e-12)

    def test_outer_audit_opt_in_compares_three_policies(self):
        d=panel()
        def pred(prefix,metric,h,strategy):
            return prefix.metric(metric)[:,-1],{'name':strategy}
        with patch.object(walk_forward_audit,'automatic_forecast',side_effect=pred):
            rows=walk_forward_audit.run_walk_forward(d,'volume_kg',2,include_adaptive=True)
        self.assertEqual(len(rows),24)
        self.assertEqual({r['strategy'] for r in rows},
                         {'global','hierarchical','adaptive','last_observation'})
        self.assertEqual(len(walk_forward_audit.summarize(rows)),12)


if __name__=='__main__':
    unittest.main()
