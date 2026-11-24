"""Independent, synthetic-only checks of the nested policy evaluator."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app_core import Dataset, DIMS
from nested_backtest import outer_origins, report, historical_panel


def panel(periods=24):
    meta = np.array([['City','Co','A','Sub','Cat','Group','Leaf','Other'],
                     ['City','Co','B','Sub','Cat','Group','Leaf','Other']])
    months = [f'{2023+i//12}-{i%12+1:02d}' for i in range(periods)]
    vol = np.zeros((2, periods))
    vol[0] = 10.
    vol[0,18:21] = 20.
    vol[0,21:24] = 40.
    vol[1,21:24] = 10.
    return Dataset(meta, vol, vol*10, months, 'synthetic', dimension_names=DIMS)


class NestedBacktestTests(unittest.TestCase):
    def test_origins_are_recent_complete_disjoint_and_bounded(self):
        self.assertEqual(outer_origins(20), [])
        self.assertEqual(outer_origins(21), [17])
        self.assertEqual(outer_origins(24), [17,20])
        self.assertEqual(outer_origins(36), [17,20,23,26,29,32])
        self.assertEqual(outer_origins(60), [41,44,47,50,53,56])

    def test_pooled_errors_and_real_reselection_for_both_metrics(self):
        d = panel()
        calls = []
        def replay(prefix, metric, horizon, strategy):
            # Both metrics have only an observed A at each origin. B's future
            # identity and values are unavailable to the production selector.
            self.assertEqual(len(prefix.meta), 1)
            self.assertEqual(prefix.volume.shape[1], len(prefix.months))
            self.assertIn(len(prefix.months), (18,21))
            self.assertEqual(prefix.volume[0,-1], 10 if len(prefix.months)==18 else 20)
            self.assertFalse(np.shares_memory(prefix.volume,d.volume))
            calls.append((metric, len(prefix.months), horizon, strategy))
            prediction = prefix.metric(metric)[:,-1] * (2 if strategy=='hierarchical' else 1)
            return prediction, {'name': strategy+'-'+str(len(prefix.months))}
        with tempfile.TemporaryDirectory() as folder, patch('nested_backtest.automatic_forecast', side_effect=replay):
            for metric in ('volume_kg','value_1000_toman'):
                dest=Path(folder)/(metric+'_summary.csv')
                details=Path(folder)/(metric+'_by_group.csv')
                rows=report(d,metric,dest,details)
                self.assertEqual(len(rows),3)
                for row in rows:
                    self.assertEqual(row['origin_count'],2)
                    # Actual 20+50=70; Global errors 10+20+10=40;
                    # hierarchical errors 0+0+10=10 (unseen launch).
                    self.assertAlmostEqual(row['global_wape'],40/70)
                    self.assertAlmostEqual(row['hierarchical_wape'],10/70)
                    self.assertAlmostEqual(row['baseline_wape'],40/70)
                    self.assertAlmostEqual(row['relative_reduction_vs_baseline'],.75)
                with (Path(folder)/(metric+'_by_origin.csv')).open(encoding='utf-8-sig') as stream:
                    folds=list(csv.DictReader(stream))
                self.assertEqual(len(folds),6)
                self.assertEqual(folds[0]['hierarchical_recipe'],'hierarchical-18')
                self.assertEqual(folds[3]['hierarchical_recipe'],'hierarchical-21')
                self.assertEqual(folds[-1]['target'],d.months[23])
                with details.open(encoding='utf-8-sig') as stream:
                    group=list(csv.DictReader(stream))
                market=[x for x in group if x['level']=='Market' and x['horizon']=='1'][0]
                self.assertAlmostEqual(float(market['hierarchical_wape']),10/70)
                self.assertEqual(len([x for x in group if x['level']=='Series']),6)
        self.assertEqual(len(set(calls)),24)
        self.assertEqual(len(calls),24)

    def test_future_mutation_does_not_change_historical_panel(self):
        d=panel()
        a, known=historical_panel(d,17)
        d.volume[:,18:]=999
        d.value[:,18:]=99999
        b, other=historical_panel(d,17)
        np.testing.assert_array_equal(known,other)
        np.testing.assert_array_equal(a.volume,b.volume)
        np.testing.assert_array_equal(a.value,b.value)

    def test_actual_production_engine_on_constant_panel(self):
        import auto_engine
        d = panel(21)
        d.volume[0,:]=10.
        d.value[0,:]=100.
        d.volume[1,:]=0.
        d.value[1,:]=0.
        with tempfile.TemporaryDirectory() as folder, patch.object(auto_engine,'OUT',Path(folder)):
            for metric in ('volume_kg','value_1000_toman'):
                rows=report(d,metric,Path(folder)/(metric+'_summary.csv'))
                for row in rows:
                    self.assertEqual(row['origin_count'],1)
                    self.assertEqual(row['baseline_wape'],0.)
                    self.assertEqual(row['global_wape'],0.)
                    self.assertEqual(row['hierarchical_wape'],0.)

    def test_zero_actual_is_undefined_not_perfect(self):
        d=panel(21)
        d.volume[:,18:]=0
        d.value[:,18:]=0
        with tempfile.TemporaryDirectory() as folder, patch('nested_backtest.automatic_forecast',
                return_value=(np.array([10.]), {'name':'synthetic'})):
            rows=report(d,'volume_kg',Path(folder)/'summary.csv')
        self.assertIsNone(rows[0]['hierarchical_wape'])
        self.assertEqual(rows[0]['hierarchical_absolute_error'],10.)

    def test_short_history_does_not_invent_outer_score(self):
        with tempfile.TemporaryDirectory() as folder, patch('nested_backtest.automatic_forecast') as predict:
            rows=report(panel(20),'volume_kg',Path(folder)/'summary.csv')
            self.assertFalse(predict.called)
        self.assertTrue(all(r['origin_count']==0 and r['global_wape'] is None for r in rows))


if __name__ == '__main__':
    unittest.main()
