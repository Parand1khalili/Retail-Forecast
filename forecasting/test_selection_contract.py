"""Candidate registry contract and real recipe-competition regression tests.

Synthetic panels only; no confidential EMRC workbook is required.
"""
import unittest
import numpy as np
from selection_core import eligible_candidates
from auto_engine import select_recipes


class SyntheticPanel:
    def __init__(self):
        self.meta=np.array([
            ['All', 'A'], ['All', 'B'], ['All', 'C'], ['All', 'D']
        ],dtype=object)
        self.months=[f'period-{j}' for j in range(30)]
        self.volume=np.full((4,30),10.,dtype=float)
        self.periodicity='monthly'

    def metric(self,metric):
        return self.volume

    def all_dimensions(self):
        return ['Market','SKU']


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.d=SyntheticPanel()
        self.origins=[17,20,23]

    def grid(self):
        return {o:{
            'base|A':np.full((4,3),13.),
            'ml|raw|24':np.full((4,3),10.),
        } for o in self.origins}

    def test_candidate_incompatible_metric_is_excluded(self):
        grid=self.grid()
        for origin in self.origins:
            grid[origin]['price-trend|2|0.5|Mean 2']=np.full((4,3),10.)
        accepted,rejected=eligible_candidates(
            grid,self.origins,'volume_kg','monthly',4)
        self.assertEqual(accepted,{'base|A','ml|raw|24'})
        self.assertEqual(rejected['price-trend|2|0.5|Mean 2'],
                         'incompatible metric or periodicity')

    def test_incomplete_invalid_and_unregistered_candidates_are_audited(self):
        grid=self.grid()
        grid[self.origins[0]]['ml|scaled|all']=np.full((4,3),10.)
        for origin in self.origins:
            grid[origin]['ml|log|all']=np.full((4,3),float('nan'))
            grid[origin]['unknown|model']=np.full((4,3),10.)
        grid[self.origins[-1]]['base|A']=np.full((3,3),13.)
        accepted,rejected=eligible_candidates(grid,self.origins,'volume_kg','monthly',4)
        self.assertEqual(accepted,{'ml|raw|24'})
        self.assertIn('not available',rejected['ml|scaled|all'])
        self.assertIn('invalid shape',rejected['base|A'])
        self.assertIn('invalid shape',rejected['ml|log|all'])
        self.assertIn('unregistered',rejected['unknown|model'])

    def test_insufficient_history_is_rejected(self):
        grid={11:{'base|Mean 2|2':np.ones((4,3)),
                  'ml|raw|24':np.ones((4,3))}}
        accepted,rejected=eligible_candidates(grid,[11],'volume_kg','monthly',4)
        self.assertEqual(accepted,{'base|Mean 2|2'})
        self.assertIn('insufficient history',rejected['ml|raw|24'])

    def test_real_selector_picks_stable_candidate_all_horizons(self):
        policy,records=select_recipes(self.d,'volume_kg',self.origins,self.grid())
        self.assertEqual(len(policy),3)
        self.assertTrue(all(policy[str(h)]['name']=='ml|raw|24' for h in (1,2,3)))
        self.assertTrue(all(policy[str(h)]['validation_wape']==0. for h in (1,2,3)))
        self.assertEqual(len(records),6)
        self.assertEqual(policy['1']['selection_evidence']['rejected_candidates'],{})

    def test_future_actuals_do_not_change_past_competition(self):
        grid=self.grid()
        before,_=select_recipes(self.d,'volume_kg',self.origins,grid)
        self.d.volume[:,27:]=999999.
        after,_=select_recipes(self.d,'volume_kg',self.origins,grid)
        self.assertEqual(before,after)

    def test_global_winner_differs_from_noisy_legacy_candidate(self):
        grid=self.grid()
        for origin in self.origins:
            grid[origin]['ml|raw|24']=np.full((4,3),14.)
        # One single surprising month cannot promote a challenger.
        grid[self.origins[0]]['ml|raw|24']=np.full((4,3),10.)
        guarded,_=select_recipes(self.d,'volume_kg',self.origins,grid)
        legacy,_=select_recipes(self.d,'volume_kg',self.origins,grid,
                                selection_mode='legacy')
        self.assertEqual(guarded['1']['name'],'base|A')
        self.assertEqual(legacy['1']['name'],'ml|raw|24')


if __name__=='__main__':
    unittest.main()
