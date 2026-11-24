"""Outer policy replay: independent expectations, cutoff isolation and live path."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_core import Dataset
import auto_engine
import walk_forward_audit as audit


def panel(frequency='monthly'):
    meta=np.array([['Tehran','Co',b,'Sub','C','Not specified','Not specified','Not specified']
                   for b in ('A','B','New')])
    q=np.vstack([np.arange(1.,25.),np.full(24,2.),np.zeros(24)])
    q[2,18:]=4
    price=np.ones(24)*2
    price[20:]=4
    dates=[str(p.asfreq('M','end')) for p in pd.period_range(
        '2023-01',periods=24,freq='2M' if frequency=='bimonthly' else 'M')]
    return Dataset(meta,q,q*price,dates,'synthetic',periodicity=frequency)


class WalkForwardTests(unittest.TestCase):
    def test_origin_count_and_disjoint_target_blocks(self):
        self.assertEqual(audit.outer_origins(24),[17,20])
        self.assertEqual(audit.outer_origins(43),[33,36,39])
        with self.assertRaises(ValueError):audit.outer_origins(23)
        with self.assertRaises(ValueError):audit.outer_origins(43,1)

    def test_full_policy_receives_only_prefix_and_scores_new_series(self):
        for frequency in ('monthly','bimonthly'):
            d=panel(frequency)
            calls=[]
            def predict(prefix,metric,h,strategy):
                calls.append((len(prefix.months),len(prefix.meta),h,strategy))
                return prefix.metric(metric)[:,-1],{'name':'controlled'}
            with patch.object(audit,'automatic_forecast',side_effect=predict):
                rows=audit.run_walk_forward(d,'volume_kg',2)
            self.assertEqual(len(calls),12)
            self.assertEqual(len(rows),18)
            self.assertEqual(calls[:2],[(18,2,1,'global'),(18,2,1,'hierarchical')])
            self.assertEqual(calls[-2:],[(21,3,3,'global'),(21,3,3,'hierarchical')])
            r=rows[0]
            self.assertEqual(r['target'],d.months[18])
            self.assertEqual(r['actual'],25)
            self.assertEqual(r['predicted'],20)
            self.assertEqual(r['unobserved_actual'],4)
            self.assertEqual(r['cells'],3)
            self.assertAlmostEqual(r['wape'],5/25)
            self.assertAlmostEqual(r['mae'],5/3)
            self.assertAlmostEqual(r['bias'],-5/25)
            summary=next(r for r in audit.summarize(rows) if r['strategy']=='global' and r['horizon']==1)
            # Second fold: actual 28, predicted 27, absolute error 1.
            self.assertEqual(summary['actual'],53)
            self.assertEqual(summary['absolute_error'],6)
            self.assertAlmostEqual(summary['wape'],6/53)
            self.assertAlmostEqual(summary['market_wape'],6/53)

    def test_future_target_changes_scores_not_earlier_predictions(self):
        d=panel()
        def predict(prefix,metric,h,strategy):
            return prefix.metric(metric)[:,-1],{'name':'controlled'}
        with patch.object(audit,'automatic_forecast',side_effect=predict):
            first=audit.run_walk_forward(d,'volume_kg',2)
            d.volume[:,18:]+=100
            second=audit.run_walk_forward(d,'volume_kg',2)
        before=[r for r in first if r['origin']==d.months[17]]
        after=[r for r in second if r['origin']==d.months[17]]
        self.assertEqual([r['predicted'] for r in before],[r['predicted'] for r in after])
        self.assertNotEqual([r['actual'] for r in before],[r['actual'] for r in after])

    def test_actual_engine_replays_price_guard_and_ordinary_policy(self):
        d=panel()
        # Only the expensive model fit is replaced; the live entry point,
        # candidate construction, selection, Price Guard and application run.
        def fitted(prefix,metric,cutoff,window,predict_only=None):
            last=prefix.metric(metric)[:,cutoff]
            return {'raw':np.repeat(last[:,None],3,axis=1)}
        with tempfile.TemporaryDirectory() as folder, patch.object(auto_engine,'OUT',Path(folder)), \
             patch.object(auto_engine,'fit_models',side_effect=fitted):
            rows=audit.run_walk_forward(d,'value_1000_toman',2)
        first=[r for r in rows if r['origin']==d.months[17] and r['strategy']!='last_observation']
        second=[r for r in rows if r['origin']==d.months[20] and r['strategy']!='last_observation']
        self.assertTrue(all(not r['price_guard_active'] for r in first))
        self.assertTrue(all(r['price_guard_active'] for r in second))
        self.assertTrue(all(r['recipe']=='robust price-shift forecast' for r in second))
        self.assertTrue(all(r['predicted']==108 for r in second))


if __name__=='__main__':
    unittest.main()
