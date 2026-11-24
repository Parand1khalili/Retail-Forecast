"""Synthetic-only Model Lab tests: fair summaries, bounded previews, safe exports."""
import csv
from io import BytesIO, StringIO
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

import backtest_lab


class FakeDataset:
    def __init__(self, periods=36, markets=('All',)):
        self.meta = np.asarray([[market, 'Brand X'] for market in markets], dtype=object)
        self.months = [f'{2023+i//12}-{1+i%12:02d}' for i in range(periods)]
        self.periodicity = 'monthly'
        self.source = 'test_master.xlsx'

    def all_dimensions(self):
        return ['Market', 'Brand']

    def report_grain_levels(self):
        return [None]


def fake_report(dataset, metric, summary, detailed_output=None, holdout=True):
    rows = []
    for h in (1, 2, 3):
        rows.append({
            'metric': metric, 'horizon': str(h),
            'audit_type': 'untouched_holdout' if holdout else 'no_independent_holdout',
            'origin': '2025-08' if holdout else '',
            'selection_origins': '2025-02,2025-05',
            'global_recipe': 'base|Mean 2|2',
            'legacy_global_recipe': 'base|Last month|2',
            'legacy_global_wape': .2 if holdout else None,
            'global_wape': .15 if holdout else None,
            'hierarchical_wape': .1 if holdout else None,
            'relative_reduction_vs_legacy': .5 if holdout else None,
            'selection_only': not holdout,
        })
    with summary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if holdout:
        values = []
        for level in ('Subcategory', 'Brand', 'SKU', 'Series'):
            for i in range(26):
                values.append({
                    'metric':metric,'horizon':'1','level':level,
                    'group':f'{level} {i}', 'series':1,
                    'actual':i, 'global_wape':.15, 'hierarchical_wape':.1,
                    'legacy_global_wape':.2, 'relative_reduction_vs_legacy':.5,
                    'improvement_vs_legacy':'improved' if i%2 else 'deteriorated',
                    'global_mae':2., 'hierarchical_mae':1.,
                    'global_bias':0., 'hierarchical_bias':0.,
                })
        with detailed_output.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=values[0])
            writer.writeheader()
            writer.writerows(values)
    return rows


class BacktestLabTests(unittest.TestCase):
    def test_both_metrics_holdout_and_full_csv_zip(self):
        metrics=['volume_kg','value_1000_toman']
        with patch.object(backtest_lab, 'report', side_effect=fake_report) as audit:
            result, contents=backtest_lab.run_backtest(FakeDataset(),metrics)
        self.assertEqual(audit.call_count,2)
        self.assertEqual(len(result['summaries']),6)
        self.assertTrue(result['holdout_available'])
        self.assertEqual(result['market'],'All')
        self.assertEqual(len(result['group_preview']),2*3*15)
        self.assertFalse(any(row['level']=='Series' for row in result['group_preview']))
        groups=[g for g in result['group_totals'] if g['level']=='SKU']
        self.assertEqual(len(groups),2)
        self.assertTrue(all(g['groups']==26 for g in groups))
        selected=[x for x in result['group_preview']
                  if x['metric']=='volume_kg' and x['level']=='SKU']
        self.assertEqual(len(selected),15)
        self.assertEqual(min(float(x['actual']) for x in selected),11.)
        with ZipFile(BytesIO(contents)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()),4)
            self.assertIn('volume_kg_by_group.csv',archive.namelist())
            with archive.open('volume_kg_by_group.csv') as stream:
                text=stream.read().decode('utf-8-sig')
            self.assertIn('Series 25',text)
            self.assertIn('SKU 0',text)

    def test_without_independent_holdout_no_accuracy_is_invented(self):
        def no_holdout(*args, **kwargs):
            return fake_report(*args, **kwargs, holdout=False)
        with patch.object(backtest_lab, 'report', side_effect=no_holdout):
            result, contents=backtest_lab.run_backtest(FakeDataset(),['volume_kg'])
        self.assertFalse(result['holdout_available'])
        self.assertEqual(result['group_preview'],[])
        self.assertEqual(result['group_totals'],[])
        self.assertTrue(all(row['hierarchical_wape'] is None
                            for row in result['summaries']))
        with ZipFile(BytesIO(contents)) as archive:
            self.assertEqual(archive.namelist(),['volume_kg_summary.csv'])

    def test_invalid_metrics_are_rejected_before_model_training(self):
        for bad in ([], ['bad_metric'], ['volume_kg','volume_kg'], 'volume_kg', None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                backtest_lab.run_backtest(FakeDataset(),bad)

    def test_ineligible_short_history_and_overlapping_market_rejected(self):
        with self.assertRaisesRegex(ValueError,'۱۸'):
            backtest_lab.run_backtest(FakeDataset(periods=8),['volume_kg'])
        with self.assertRaisesRegex(ValueError,'Market'):
            backtest_lab.run_backtest(FakeDataset(markets=('All','City')),['volume_kg'])


class RealPolicyIntegrationTests(unittest.TestCase):
    def test_actual_policy_and_holdout_pipeline_to_downloadable_group_reports(self):
        """Only the costly candidate generation is replaced; policy and audit are real."""
        import tempfile
        from pathlib import Path
        from app_core import Dataset, DIMS
        import auto_engine

        dims = DIMS + ['Market', 'SKU']
        meta = []
        for label in ('A', 'B'):
            for i in range(8):
                meta.append(['Tehran', 'Company', label + str(i), 'Subbrand',
                             label, 'Group', 'Leaf', 'Other', 'All', 'sku-' + str(i)])
        months = [f'{2023+i//12}-{1+i%12:02d}' for i in range(36)]
        d = Dataset(np.array(meta, dtype='<U36'),
                    np.full((16, 36), 20.),
                    np.full((16, 36), 200.),
                    months, 'synthetic-master', dimension_names=dims)

        def fake_candidates(dataset, metric, origin):
            n = len(dataset.meta)
            base = np.full((n, 3), 22.)
            a = np.full((n, 3), 20.)
            b = np.full((n, 3), 20.)
            group_b = dataset.meta[:, DIMS.index('Subcategory')] == 'B'
            a[group_b] = 27.
            b[~group_b] = 30.
            return {'base|Mean 2|2': base, 'ml|raw|24': a,
                    'ml|scaled|24': b}

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(auto_engine, 'OUT', Path(folder)), \
             patch.object(auto_engine, 'candidates', side_effect=fake_candidates), \
             patch.object(auto_engine, 'forecast_component',
                          side_effect=lambda dataset, metric, origin, name: fake_candidates(dataset, metric, origin)[name]):
            response, zip_bytes = backtest_lab.run_backtest(d, ['volume_kg'])
        self.assertEqual(len(response['summaries']), 3)
        self.assertTrue(response['holdout_available'])
        self.assertTrue(any(x['level']=='SKU' for x in response['group_totals']))
        self.assertTrue(any(x['level']=='Subcategory' for x in response['group_totals']))
        for item in response['summaries']:
            self.assertEqual(item['audit_type'], 'nested_walk_forward')
            self.assertEqual(item['origin_count'], 6)
            self.assertAlmostEqual(item['global_wape'], .1)
            self.assertAlmostEqual(item['hierarchical_wape'], .1*4/6)
            self.assertEqual(item['baseline_wape'], 0.)
            self.assertIsNone(item['relative_reduction_vs_baseline'])
        with ZipFile(BytesIO(zip_bytes)) as archive:
            self.assertIsNone(archive.testzip())
            details = archive.read('volume_kg_by_group.csv').decode('utf-8-sig')
            self.assertIn('volume_kg_by_origin.csv', archive.namelist())
        self.assertIn('Subcategory', details)
        self.assertIn('SKU', details)


if __name__=='__main__':
    unittest.main()
