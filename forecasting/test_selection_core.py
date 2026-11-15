"""Pure, independently reproducible validation-policy tests."""
import unittest
import numpy as np

from selection_core import (rolling_plan, BacktestConfig, ChampionGuard,
                            choose_champion, candidate_registry, candidate_info)


class ValidationCoreTests(unittest.TestCase):
    def test_no_overlap_and_contiguous_holdout(self):
        for cutoff in (29, 30, 35, 48):
            plan = rolling_plan(cutoff)
            self.assertTrue(plan['independent'])
            self.assertGreaterEqual(len(plan['selection']), 2)
            self.assertLess(plan['selection'][-1]+3, plan['holdout']+1)
            self.assertLessEqual(plan['holdout']+3, cutoff)
            self.assertEqual(plan['selection'], sorted(set(plan['selection'])))

    def test_short_history_and_global_are_explicit_legacy_fallback(self):
        self.assertIsNone(rolling_plan(24)['holdout'])
        self.assertIsNone(rolling_plan(35, strategy='global')['holdout'])
        self.assertIsNone(rolling_plan(35, categories_available=False)['holdout'])

    def test_bimonthly_origins_are_observation_indices(self):
        plan = rolling_plan(35)
        self.assertEqual(plan['selection'], [23, 26, 29])
        self.assertEqual(plan['holdout'], 32)

    def test_bad_config_rejected(self):
        with self.assertRaises(ValueError):
            rolling_plan(35, config=BacktestConfig(selection_stride=2))

    def test_candidate_registry_is_unique_and_metrics_are_constrained(self):
        catalog = candidate_registry(['ml|raw|24', 'base|Mean 2|2',
                                      'ml|raw|24', 'price-trend|2|0.5|Mean 2'])
        self.assertEqual(len(catalog), 3)
        self.assertEqual(candidate_info('price-trend|2|0.5|Mean 2').metrics,
                         ('value_1000_toman',))
        self.assertEqual(candidate_info('ml|raw|24').min_history, 18)
        with self.assertRaises(ValueError):
            candidate_registry(['nonsense|candidate'])

    def test_stable_candidate_wins(self):
        winner, report = choose_champion({'base|A': [10,10,10],
                                          'ml|raw|24': [7,7,7]}, 'base|A')
        self.assertEqual(winner,'ml|raw|24')
        self.assertEqual(report['legacy_best'],winner)

    def test_noisy_challenger_cannot_win_with_one_sharp_spike(self):
        winner, report = choose_champion({'base|A': [10,10,10],
                                          'ml|raw|24': [0,11,11]}, 'base|A')
        self.assertEqual(report['legacy_best'],'ml|raw|24')
        self.assertEqual(winner,'base|A')

    def test_all_zero_reference_does_not_promote(self):
        winner, report = choose_champion({'base|A': [0,0,0],
                                          'ml|raw|24': [0,0,0]}, 'base|A')
        self.assertEqual(winner,'base|A')

    def test_not_equal_target_lengths_rejected(self):
        with self.assertRaises(ValueError):
            choose_champion({'base|A':[1,2], 'ml|raw|24':[1]},'base|A')


if __name__=='__main__':
    unittest.main()
