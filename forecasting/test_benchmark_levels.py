"""Synthetic regression tests for independent group-level benchmarking."""
import unittest
import numpy as np
from benchmark_levels import compare_levels, error_metrics, summarize_levels


class D:
    def __init__(self):
        self.meta=np.asarray([
            ['All', 'Drinks', 'Juice', 'Brand A', 'sku1'],
            ['All', 'Drinks', 'Juice', 'Brand A', 'sku2'],
            ['All', 'Snacks', 'Crackers', 'Brand B', 'sku3'],
            ['All', 'Snacks', 'Crackers', 'Brand B', 'sku4']],dtype=object)
        self.volume=np.zeros((4,36))
        self.volume[0,22:28]=10
        self.volume[1,22:28]=10

    def all_dimensions(self):
        return ['Market','Subcategory','Subcategory2','Brand','SKU']

    def metric(self,name):
        return self.volume


class BenchmarkTests(unittest.TestCase):
    def test_independent_group_scoring_has_no_double_count(self):
        d=D()
        actual=np.array([10.,10.,0.,0.])
        ref=np.array([8.,13.,5.,0.])
        new=np.array([10.,11.,0.,0.])
        rows=compare_levels(d,actual,ref,new,26)
        market=[r for r in rows if r['level']=='Market']
        self.assertEqual(len(market),1)
        self.assertEqual(market[0]['actual'],20)
        self.assertAlmostEqual(market[0]['hierarchical_wape'],.05)
        self.assertAlmostEqual(market[0]['aggregated_hierarchical_wape'],.05)
        self.assertEqual(sum(r['series'] for r in rows if r['level']=='Subcategory'),4)
        self.assertEqual(sum(r['actual'] for r in rows if r['level']=='Subcategory'),20)
        self.assertEqual(sum(r['actual'] for r in rows if r['level']=='Brand'),20)

    def test_zero_demand_group_wape_is_undefined(self):
        d=D()
        rows=compare_levels(d,np.array([10.,10.,0.,0.]),
                            np.zeros(4),np.zeros(4),26)
        snack=[r for r in rows if r['level']=='Subcategory' and r['group']=='Snacks'][0]
        self.assertIsNone(snack['hierarchical_wape'])
        self.assertEqual(snack['zero_actual_cells'],2)

    def test_sparse_segment_uses_only_pre_origin_sales(self):
        d=D()
        d.volume[2,27:]=100000
        rows=compare_levels(d,np.array([10.,10.,0.,0.]),np.zeros(4),np.zeros(4),26)
        sparse=[r for r in rows if r['level']=='Demand pattern']
        zero=[r for r in sparse if r['group']=='Zero recent volume'][0]
        self.assertEqual(zero['series'],2)

    def test_error_validation_and_breakdown(self):
        with self.assertRaises(ValueError):
            error_metrics([1.],[1.,2.])
        self.assertIsNone(error_metrics([0.],[7.])['wape'])
        summary=summarize_levels(compare_levels(D(),np.array([10.,10.,0.,0.]),
                                              np.zeros(4),np.zeros(4),26))
        self.assertEqual(summary['Subcategory']['groups'],2)


if __name__=='__main__':
    unittest.main()
