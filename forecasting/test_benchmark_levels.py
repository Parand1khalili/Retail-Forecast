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

    def test_sku_identity_is_aggregated_across_cities_without_brand_collision(self):
        class NamedDataset:
            def __init__(self):
                self.meta=np.asarray([
                    ['All','Tehran','Company A','Shared','Item 1'],
                    ['All','Shiraz','Company A','Shared','Item 1'],
                    ['All','Tehran','Company B','Shared','Item 1'],
                ],dtype=object)
                self.volume=np.ones((3,36))
            def all_dimensions(self):
                return ['Market','City','Company','Brand','SKU']
            def metric(self,metric):
                return self.volume
        d=NamedDataset()
        results=compare_levels(d,np.array([10.,20.,30.]),
                               np.array([8.,20.,30.]),
                               np.array([10.,20.,30.]),26)
        sku=[item for item in results if item['level']=='SKU']
        brands=[item for item in results if item['level']=='Brand']
        self.assertEqual(len(sku),2)
        self.assertEqual(sorted(item['series'] for item in sku),[1,2])
        self.assertEqual(sorted(item['actual'] for item in sku),[30.,30.])
        self.assertEqual(len(brands),2)
        self.assertEqual(sum(item['actual'] for item in brands),60.)
        brief=compare_levels(d,np.array([10.,20.,30.]),
                             np.array([8.,20.,30.]),
                             np.array([10.,20.,30.]),26,include_series=False)
        self.assertFalse(any(item['level'] in ('SKU','Series') for item in brief))

    def test_description_is_sku_only_when_report_grain_confirms_it(self):
        class DescriptionDataset:
            def __init__(self,caption):
                self.meta=np.asarray([['All','Company','Brand',caption,'Item X']],
                                     dtype=object)
                self.volume=np.ones((1,36))
            def all_dimensions(self):
                return ['Market','Company','Brand','Element Caption','Element Description']
            def metric(self,name):
                return self.volume
        for caption,expected in (('Product','Element Description'),('SKU','SKU'),
                                 ('Ordinary brand name','Element Description')):
            d=DescriptionDataset(caption)
            rows=compare_levels(d,np.ones(1),np.ones(1),np.ones(1),26)
            levels={row['level'] for row in rows}
            self.assertIn(expected,levels)
            if expected!='SKU':
                self.assertNotIn('SKU',levels)

    def test_error_validation_and_breakdown(self):
        with self.assertRaises(ValueError):
            error_metrics([1.],[1.,2.])
        self.assertIsNone(error_metrics([0.],[7.])['wape'])
        summary=summarize_levels(compare_levels(D(),np.array([10.,10.,0.,0.]),
                                              np.zeros(4),np.zeros(4),26))
        self.assertEqual(summary['Subcategory']['groups'],2)


if __name__=='__main__':
    unittest.main()
