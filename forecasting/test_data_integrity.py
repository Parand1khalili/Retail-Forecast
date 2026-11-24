"""End-to-end Master integrity checks with independently specified expected totals."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

import numpy as np

import app_core
from app_core import Dataset
from schema import SchemaError, resolve_fields


def workbook(path, sheets):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'+
            ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(sheets,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+
            ''.join(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+'</Relationships>')
        for i,rows in enumerate(sheets.values(),1):
            xml=[]
            for j,row in enumerate(rows,1):
                cells=[]
                for k,val in enumerate(row):
                    if val is not None:
                        cells.append(f'<c r="{chr(65+k)}{j}" t="inlineStr"><is><t>{escape(str(val))}</t></is></c>')
                xml.append(f'<row r="{j}">'+''.join(cells)+'</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="{ns}"><sheetData>'+''.join(xml)+'</sheetData></worksheet>')


def sales(dates=('2026-01','2026-02'), volumes=(10,20), values=(100,200),
          market='City', brand='A', size='Small', grain='SKU', unit='Kg'):
    return [['Market','City','Brand','Element Caption','Size','Measure',*dates],
            [market,'Tehran',brand,grain,size,'Volume Sales ('+unit+')',*volumes],
            [market,'Tehran',brand,grain,size,'Value Sales (Toman)',*values]]


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        p=patch.object(app_core,'CACHE',self.root);p.start();self.addCleanup(p.stop)
        self.path=self.root/'master.xlsx'

    def load(self,sheets,mapping=None):
        workbook(self.path,sheets)
        return Dataset.file(self.path,mapping)

    def test_identical_overlap_and_disjoint_periods_reconcile(self):
        d=self.load({'old':sales(), 'new':sales(('2026-02','2026-03'),(20,30),(200,300)),
                     'Notes':[['Documentation only']]})
        np.testing.assert_array_equal(d.volume,[[10,20,30]])
        np.testing.assert_array_equal(d.value,[[100,200,300]])
        audit=d.schema['integrity']
        self.assertEqual(audit['status'],'passed')
        self.assertEqual(len(audit['identical_overlaps']),1)
        self.assertEqual(audit['skipped_sheets'][0]['sheet'],'Notes')
        r=audit['reconciliation'][0]
        self.assertEqual(r['source_volume'],[10,20,30])
        self.assertEqual(r['imported_value'],[100,200,300])
        self.assertEqual(r['max_absolute_difference'],[0,0])
        self.assertEqual(Dataset.file(self.path).schema,d.schema)

    def test_conflicting_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'متعارض'):
            self.load({'old':sales(),'new':sales(('2026-02','2026-03'),(99,30),(200,300))})

    def test_absent_series_zero_is_explicitly_audited(self):
        first=sales();first+=sales(brand='B')[1:]
        d=self.load({'old':first,'new':sales(('2026-03','2026-04'))})
        np.testing.assert_array_equal(d.volume,[[10,20,10,20],[10,20,0,0]])
        r=d.schema['integrity']['reconciliation'][0]
        self.assertEqual(r['source_volume'],[20,40,10,20])
        self.assertEqual(r['inferred_zero_series_by_period'],[0,0,1,1])
        self.assertEqual(d.schema['integrity']['inferred_zero_series_periods'],2)

    def test_complementary_same_period_sheets_preserve_distinct_rows(self):
        d=self.load({'A':sales(), 'B':sales(brand='B',volumes=(3,7),values=(30,70))})
        np.testing.assert_array_equal(d.volume,[[10,20],[3,7]])
        np.testing.assert_array_equal(d.value,[[100,200],[30,70]])
        r=d.schema['integrity']['reconciliation'][0]
        self.assertEqual(r['source_volume'],[13,27])
        self.assertEqual(r['inferred_zero_series_periods'],0)
        self.assertEqual(d.schema['integrity']['identical_overlaps'],[])

    def test_partial_row_overlap_deduplicates_before_projection(self):
        first=sales();first+=sales(size='Large',volumes=(2,3),values=(20,30))[1:]
        second=sales();second+=sales(size='Medium',volumes=(4,5),values=(40,50))[1:]
        d=self.load({'one':first,'two':second},{'dimension_columns':['City','Brand','Element Caption']})
        np.testing.assert_array_equal(d.volume,[[16,28]])
        r=d.schema['integrity']['reconciliation'][0]
        self.assertEqual(r['raw_sheet_volume'],[26,48])
        self.assertEqual(r['duplicate_volume'],[10,20])
        self.assertEqual(r['source_volume'],[16,28])
        self.assertEqual(sum(x['series_count'] for x in d.schema['integrity']['identical_overlaps']),2)

    def test_partial_row_overlap_conflict_is_not_overwritten(self):
        first=sales();first+=sales(brand='B')[1:]
        second=sales(volumes=(99,20));second+=sales(brand='C')[1:]
        with self.assertRaisesRegex(ValueError,'متعارض'):
            self.load({'old':first,'new':second})

    def test_explicit_missing_period_is_not_inferred_zero(self):
        first=sales(('2026-01','2026-02','2026-03'),(None,20,30),(None,200,300))
        second=sales(('2026-01','2026-02','2026-03'),(1,2,3),(10,20,30),brand='B')
        with self.assertRaisesRegex(ValueError,'مفقود'):
            self.load({'one':first,'two':second})
        # Another sheet may supply the exact missing identity and period.
        third=sales(('2026-01','2026-02'),(10,20),(100,200))
        d=self.load({'one':first,'two':second,'three':third})
        np.testing.assert_array_equal(d.volume,[[10,20,30],[1,2,3]])
        self.assertEqual(d.schema['integrity']['inferred_zero_series_periods'],0)

    def test_new_series_zero_before_first_observation(self):
        d=self.load({'early':sales(), 'late':sales(('2026-03','2026-04'),brand='B')})
        np.testing.assert_array_equal(d.volume,[[10,20,0,0],[0,0,10,20]])
        self.assertEqual(d.schema['integrity']['inferred_zero_series_periods'],4)

    def test_disjoint_markets_separated_and_incomplete_scope_rejected(self):
        d=self.load({'city':sales(),'national':sales(market='All',volumes=(100,200))})
        self.assertEqual(d.for_market('City').volume.sum(),30)
        self.assertEqual(d.for_market('All').volume.sum(),300)
        self.assertEqual(len(d.schema['integrity']['reconciliation']),2)
        with self.assertRaisesRegex(ValueError,'پوشش دوره'):
            self.load({'city':sales(),'national':sales(('2026-03','2026-04'),market='All')})

    def test_duplicate_raw_rows_even_after_projection(self):
        data=sales();data.append(data[1].copy())
        with self.assertRaisesRegex(ValueError,'2 و 4 کاملاً تکراری'):
            self.load({'sales':data},{'dimension_columns':['City','Brand','Element Caption']})

    def test_distinct_rows_with_same_identity_are_aggregated_and_audited(self):
        rows=sales()
        rows+=sales(volumes=(5,7),values=(50,70))[1:]
        d=self.load({'sales':rows})
        np.testing.assert_array_equal(d.volume,[[15,27]])
        np.testing.assert_array_equal(d.value,[[150,270]])
        audit=d.schema['integrity']['sheets'][0]
        self.assertEqual(audit['source_sales_rows'],4)
        self.assertEqual(audit['aggregated_distinct_rows'],2)
        self.assertEqual(audit['multirow_measure_identities'],2)
        self.assertEqual(audit['aggregation_examples'][0]['row'],'4')
        self.assertEqual(d.schema['integrity']['reconciliation'][0]['source_volume'],[15,27])
        self.assertEqual(d.schema['integrity']['reconciliation'][0]['max_absolute_difference'],[0,0])

    def test_equal_forecast_values_with_distinct_other_source_cells_preserved(self):
        rows=sales()
        rows[0].append('Year-2025')
        rows[1].append(100)
        rows[2].append(1000)
        rows.append(rows[1][:-1]+[200])
        rows.append(rows[2][:-1]+[2000])
        d=self.load({'sales':rows})
        np.testing.assert_array_equal(d.volume,[[20,40]])
        np.testing.assert_array_equal(d.value,[[200,400]])
        self.assertEqual(d.schema['integrity']['sheets'][0]['aggregated_distinct_rows'],2)

    def test_distinct_rows_with_invalid_value_are_not_ignored(self):
        rows=sales()
        rows+=sales(volumes=(5,'bad'),values=(50,70))[1:]
        with self.assertRaisesRegex(ValueError,'نامعتبر'):
            self.load({'sales':rows})

    def test_deliberate_dimension_projection_reconciles(self):
        data=sales();data+=sales(size='Large',volumes=(2,3),values=(20,30))[1:]
        d=self.load({'sales':data},{'dimension_columns':['City','Brand','Element Caption']})
        np.testing.assert_array_equal(d.volume,[[12,23]])
        self.assertEqual(d.schema['integrity']['sheets'][0]['excluded_dimensions'],['Size'])
        self.assertEqual(d.schema['integrity']['reconciliation'][0]['source_series'],2)

    def test_pairing_before_projection(self):
        data=sales();data[2][4]='Large'
        with self.assertRaisesRegex(SchemaError,'جفت نیستند'):
            self.load({'sales':data},{'dimension_columns':['City','Brand','Element Caption']})

    def test_mixed_units_rejected_even_with_override(self):
        data=sales();data+=sales(brand='B',unit='Gr')[1:]
        with self.assertRaisesRegex(ValueError,'واحد'):
            self.load({'sales':data})
        with self.assertRaisesRegex(ValueError,'واحد'):
            self.load({'sales':data},{'units':{'volume_kg':'kg'}})
        with self.assertRaisesRegex(ValueError,'واحد'):
            self.load({'one':sales(),'two':sales(unit='Gr')})

    def test_zero_and_missing_and_ignored_measure_distinguished(self):
        rows=sales(volumes=(0,20),values=(0,200))
        rows.append(['City','Tehran','A','SKU','Small','Market Share',12,20])
        d=self.load({'sales':rows})
        self.assertEqual(d.volume[0,0],0)
        self.assertEqual(d.schema['integrity']['sheets'][0]['excluded_measures'],{'Market Share':1})
        rows[1][-1]=None
        with self.assertRaisesRegex(ValueError,'مفقود'):
            self.load({'sales':rows})
        rows=sales(('2026-01','2026-02','2026-03'),(None,20,30),(None,200,300))
        d=self.load({'sales':rows})
        self.assertEqual(d.months,['2026-02','2026-03'])
        self.assertEqual(d.schema['integrity']['sheets'][0]['unavailable_periods'],['2026-01'])

    def test_bad_sales_sheet_never_skipped_if_other_sheet_valid(self):
        bad=sales();bad[0][2]='Unknown field'
        with self.assertRaises(SchemaError):
            self.load({'good':sales(),'bad':bad})
        bad=sales(('Jan-Feb 2026','Mar-Apr 2026'))
        with self.assertRaisesRegex(ValueError,'نوع دوره'):
            self.load({'good':sales(),'wrong':bad})
        bad=sales();bad[1][2]=''
        with self.assertRaisesRegex(ValueError,'Brand'):
            self.load({'good':sales(),'bad':bad})

    def test_manual_mapping_not_replaced_by_fuzzy_guess(self):
        fields,_=resolve_fields({0:'City',1:'Brand Name',2:'Measure'},{'Brand':'Missing'})
        self.assertNotIn('Brand',fields)
        with self.assertRaises(SchemaError):
            self.load({'sales':sales()},{'fields':{'Brand':'Missing'}})

    def test_mixed_grains_are_not_aggregated(self):
        data=sales();data+=sales(grain='Product')[1:]
        d=self.load({'sales':data})
        with self.assertRaisesRegex(ValueError,'سطح تجمیع'):
            d.for_grain()
        self.assertEqual(d.for_grain('SKU').volume.sum(),30)
        with self.assertRaisesRegex(SchemaError,'Element Caption'):
            self.load({'sales':data},{'dimension_columns':['City','Brand']})

    def test_unknown_unit_requires_confirmation(self):
        with self.assertRaisesRegex(SchemaError,'واحد'):
            self.load({'sales':sales(unit='')})
        d=self.load({'sales':sales(unit='')},{'units':{'volume_kg':'kg'}})
        self.assertEqual(d.volume_unit,'kg')

    def test_mixed_frequency_and_duplicate_period_rejected(self):
        with self.assertRaisesRegex(ValueError,'مخلوط'):
            self.load({'sales':sales(('2026-01','2026-02','Mar-Apr 2026'),(1,2,3),(10,20,30))})
        with self.assertRaisesRegex(ValueError,'چند ستون'):
            self.load({'sales':sales(('2026-01',"Jan' 2026"))})


if __name__=='__main__':
    unittest.main()
