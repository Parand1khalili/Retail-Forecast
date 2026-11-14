"""Integration behavior of selection, API calculation and full-row Excel forecasts."""
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import pandas as pd

from app_core import Dataset, DIMS
import auto_engine
import full_export


class HierarchicalPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        meta = []
        for category in ('A', 'B'):
            for i in range(8):
                row = ['Tehran', 'Company', f'{category}Brand{i}', 'SubBrand',
                       category, 'Child', 'Leaf', 'Other', 'All']
                meta.append(row)
        self.d = Dataset(
            np.array(meta, dtype='<U80'),
            np.full((16, 36), 20.),
            np.full((16, 36), 20.),
            [str(p) for p in pd.period_range('2023-01', periods=36, freq='M')],
            'synthetic', dimension_names=DIMS + ['Market']
        )

    def candidates(self, dataset, metric, cutoff):
        a = np.full((len(dataset.meta), 3), 20.)
        b = np.full((len(dataset.meta), 3), 20.)
        group_b = dataset.meta[:, dataset.all_dimensions().index('Subcategory')] == 'B'
        a[group_b] = 30.
        b[~group_b] = 30.
        return {'base|A': a, 'base|B': b}

    def component(self, dataset, metric, cutoff, name):
        return self.candidates(dataset, metric, cutoff)[name]

    def test_category_forecast_is_used_in_interactive_and_excel_all_horizons(self):
        with patch.object(auto_engine, 'OUT', self.root), \
             patch.object(auto_engine, 'candidates', side_effect=self.candidates), \
             patch.object(auto_engine, 'forecast_component', side_effect=self.component), \
             patch.object(full_export, 'automatic_forecast',
                          wraps=auto_engine.automatic_forecast):
            for h in (1, 2, 3):
                result = auto_engine.future_request(self.d, {
                    'metric': 'volume_kg', 'horizon': h,
                    'filters': {'Market': 'All', 'Subcategory': 'B'},
                    'strategy': 'hierarchical'})
                self.assertEqual(result['prediction'], 160.)
                self.assertEqual(result['rows'], 8)
                self.assertTrue(result['category_selection']['active'])
                self.assertFalse(result['category_selection']['holdout']['global']['wape'] is None)
                self.assertEqual(result['category_selection']['selected_groups'][0]['recipe'], 'base|B')
                reference = auto_engine.future_request(self.d, {
                    'metric': 'volume_kg', 'horizon': h,
                    'filters': {'Market': 'All', 'Subcategory': 'B'},
                    'strategy': 'global'})
                self.assertEqual(reference['prediction'], 240.)

            export = full_export._modelled_rows(self.d)
            for metric in ('volume_kg', 'value_1000_toman'):
                for h in (1, 2, 3):
                    rows = self.d.meta[:, DIMS.index('Subcategory')] == 'B'
                    self.assertAlmostEqual(export[metric][rows, h - 1].sum(), 160.)

    def test_short_history_falls_back_to_global(self):
        short = Dataset(self.d.meta, self.d.volume[:, :25], self.d.value[:, :25],
                        self.d.months[:25], 'short',
                        dimension_names=self.d.all_dimensions())
        with patch.object(auto_engine, 'OUT', self.root), \
             patch.object(auto_engine, 'candidates', side_effect=self.candidates), \
             patch.object(auto_engine, 'forecast_component', side_effect=self.component):
            _, calibration = auto_engine.automatic_forecast(short, 'volume_kg', 1)
        self.assertFalse(calibration['category_selection']['active'])
        self.assertIsNone(calibration['category_selection']['holdout'])

    def test_missing_category_stays_global(self):
        self.d.meta[:, DIMS.index('Subcategory')] = 'Not specified'
        self.d.meta[:, DIMS.index('Subcategory2')] = 'Not specified'
        self.d.meta[:, DIMS.index('Subcategory3')] = 'Not specified'
        self.d.meta[:, DIMS.index('Subcategory4')] = 'Not specified'
        with patch.object(auto_engine, 'OUT', self.root), \
             patch.object(auto_engine, 'candidates', side_effect=self.candidates), \
             patch.object(auto_engine, 'forecast_component', side_effect=self.component):
            _, calibration = auto_engine.automatic_forecast(self.d, 'volume_kg', 1)
        self.assertFalse(calibration['category_selection']['active'])

    def test_conflicting_report_grains_must_be_selected(self):
        m = np.array([['Tehran','Co','A','SKU'], ['Tehran','Co','A','Product']], dtype=str)
        d = Dataset(m,np.ones((2,36)),np.ones((2,36)),self.d.months,'grains',
                    dimension_names=['City','Company','Brand','Element Caption'])
        with self.assertRaisesRegex(ValueError,'Element Caption'):
            auto_engine.future_request(d, {'metric':'volume_kg','horizon':1,'filters':{}})
        self.assertEqual(len(d.for_grain('SKU').meta),1)


if __name__ == '__main__':
    unittest.main()
