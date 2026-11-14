"""Synthetic behavior tests for independent hierarchical recipe selection."""
import unittest
import numpy as np

from category_selection import (category_dimensions, select_category_overrides,
    series_recipes, forecast_with_recipes, holdout_comparison)


class FakeDataset:
    def __init__(self, meta, dims, sales):
        self.meta = np.array(meta, dtype=object)
        self.dims = dims
        self.volume = np.asarray(sales, float)
        self.months = [f'{y:04d}-{m:02d}' for y in range(2023, 2027) for m in range(1, 13)]

    def all_dimensions(self):
        return self.dims

    def metric(self, metric):
        return self.volume


def make_case(origins=(17, 20, 23)):
    dims = ['Market', 'Subcategory', 'Subcategory2', 'Subcategory3', 'Brand']
    meta = []
    for category in 'AB':
        for index in range(8):
            meta.append(['All', category, f'child-{index//4}',
                         f'leaf-{index//2}', f'brand-{index}'])
    dataset = FakeDataset(meta, dims, np.full((16, 48), 20.))
    forecasts = {}
    for origin in list(origins) + [27]:
        first = np.full((16, 3), 20.)
        second = np.full((16, 3), 20.)
        first[8:, :] = 30.     # Worse in category B
        second[:8, :] = 30.    # Worse in category A
        forecasts[origin] = {'base|A': first, 'base|B': second}
    selected = {str(h): {'name': 'base|A', 'components': [['base|A', 1.]],
        'selection_evidence': {'candidate_scores': [
            {'name': 'base|A', 'components': [['base|A', 1.]]},
            {'name': 'base|B', 'components': [['base|B', 1.]]}]}}
        for h in (1, 2, 3)}
    return dataset, list(origins), forecasts, selected


class CategoryRecipeTests(unittest.TestCase):
    def test_three_levels_and_nearest_parent_fallback(self):
        d, origins, predictions, selected = make_case()
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertEqual(policy['dimensions'], ['Subcategory', 'Subcategory2', 'Subcategory3'])
        self.assertEqual(policy['groups_selected'], 3)
        for horizon in (1, 2, 3):
            names = [r['name'] for r in series_recipes(d, selected[str(horizon)], policy, horizon)]
            self.assertEqual(names[:8], ['base|A'] * 8)
            self.assertEqual(names[8:], ['base|B'] * 8)

    def test_no_categories_stays_global(self):
        d, origins, predictions, selected = make_case()
        d.dims = ['Market', 'Item', 'Subcategory2', 'Subcategory3', 'Brand']
        d.meta[:, 2:4] = 'Not specified'
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertEqual(policy['dimensions'], [])
        self.assertEqual(policy['groups_selected'], 0)

    def test_only_two_category_levels(self):
        d, origins, predictions, selected = make_case()
        d.dims = ['Market', 'Subcategory', 'Subcategory2', 'Element Description', 'Brand']
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertEqual(policy['dimensions'], ['Subcategory', 'Subcategory2'])

    def test_sparse_zero_demand_inherits_parent(self):
        d, origins, predictions, selected = make_case()
        d.volume[8:, :] = 0.
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertFalse(any(node['path'][0] == 'B' for node in policy['overrides']['1']))

    def test_market_levels_cannot_be_mixed(self):
        d, origins, predictions, selected = make_case()
        d.meta[0, 0] = 'City'
        with self.assertRaisesRegex(ValueError, 'Market'):
            select_category_overrides(d, 'volume', origins, predictions, selected)

    def test_recipe_changes_numeric_predictions_and_global_mode_does_not(self):
        d, origins, predictions, selected = make_case()
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        result, usage, _ = forecast_with_recipes(d, 1, predictions[27], selected['1'], policy)
        self.assertTrue(np.array_equal(result, np.full(16, 20.)))
        self.assertEqual(usage, {'base|A': 8, 'base|B': 8})
        baseline, _, _ = forecast_with_recipes(d, 1, predictions[27], selected['1'], policy, 'global')
        self.assertTrue(np.all(baseline[8:] == 30.))

    def test_untouched_holdout_report_never_changes_selection(self):
        d, origins, predictions, selected = make_case()
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        baseline = repr(policy)
        report = holdout_comparison(d, 'volume', 27, predictions[27], selected, policy)
        self.assertEqual(baseline, repr(policy))
        self.assertEqual(report['horizons']['1']['global']['wape'], .25)
        self.assertEqual(report['horizons']['1']['hierarchical']['wape'], 0.)
        self.assertFalse(report['used_for_selection'])

    def test_unstable_origin_gains_do_not_promote_overrides(self):
        d, origins, predictions, selected = make_case()
        predictions[origins[-1]]['base|B'][8:, :] = 35.
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertFalse(any(node['path'][0] == 'B' for node in policy['overrides']['1']))

    def test_holdout_perturbation_does_not_affect_recipe_choice(self):
        d, origins, predictions, selected = make_case()
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        d.volume[:, 30:] *= 1000.  # Strictly after selection targets
        again = select_category_overrides(d, 'volume', origins, predictions, selected)
        self.assertEqual(policy, again)

    def test_fractional_forecast_and_missing_percent_denominator(self):
        d, origins, predictions, selected = make_case()
        d.volume[:, :] = 0.
        policy = select_category_overrides(d, 'volume', origins, predictions, selected)
        audit = holdout_comparison(d, 'volume', 27, predictions[27], selected, policy)
        self.assertIsNone(audit['horizons']['1']['global']['wape'])


if __name__ == '__main__':
    unittest.main()
