"""Regression checks for explicit source frequency and independent Market levels.

Run: python -m unittest discover -s forecasting -p "test_*.py" -v
These tests generate disposable XLSX files and do not contain private RA data.
"""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

import app_core
from app_core import Dataset
from schema import period, months_in_header
from auto_engine import Features, future_request


def make_xlsx(path, columns):
    headers=['Market','City/Region/Dis. Channel','Brand','Measure',
             'Year-2021','MAT-2026','YTD-2026']+columns
    rows=[headers]
    for level,location,volume in [('All','All',100),('City','Tehran',20),
                                  ('Region','North',35)]:
        for metric,amount in [('Volume Sales(Gr)',volume),
                              ('Value Sales (1,000 Toomans)',volume*10)]:
            rows.append([level,location,'Aqua',metric,'999','888','777']+
                        [str(amount+j) for j in range(len(columns))])

    def col(idx):
        x=''
        while idx:
            idx,digit=divmod(idx-1,26)
            x=chr(65+digit)+x
        return x

    xml_rows=[]
    for number,row in enumerate(rows,1):
        cells=''.join(
            f'<c r="{col(idx)}{number}" t="inlineStr"><is><t>{escape(str(val))}</t></is></c>'
            for idx,val in enumerate(row,1))
        xml_rows.append(f'<row r="{number}">{cells}</row>')
    worksheet=('<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'+''.join(xml_rows)+'</sheetData></worksheet>')
    workbook=('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sales" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels=('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
          '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
          'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
          '</Relationships>')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('xl/workbook.xml',workbook)
        z.writestr('xl/_rels/workbook.xml.rels',rels)
        z.writestr('xl/worksheets/sheet1.xml',worksheet)


class SourceFrequencyTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        p=patch.object(app_core,'CACHE',self.root)
        p.start();self.addCleanup(p.stop)

    def test_month_pairs_are_period_end_labels(self):
        self.assertEqual(period('May-Jun 2026','bimonthly'),'2026-06')
        self.assertEqual(period('Jul-Aug 2026','bimonthly'),'2026-08')
        self.assertEqual(period('Nov-Dec 2025','bimonthly'),'2025-12')
        self.assertIsNone(period('Year-2026','bimonthly'))
        self.assertIsNone(period('MAT-2026','bimonthly'))
        self.assertIsNone(period('Jan-Feb 2026','monthly'))
        self.assertIsNone(period('May-Aug 2026','bimonthly'))

    def test_all_geographic_levels_preserved_without_double_counting(self):
        path=self.root/'two_month.xlsx'
        make_xlsx(path,['Jan-Feb 2026','Mar-Apr 2026','May-Jun 2026'])
        d=Dataset.file(path,{'periodicity':'bimonthly'})
        self.assertEqual(d.months,['2026-02','2026-04','2026-06'])
        self.assertEqual(d.period_months,2)
        self.assertEqual(d.volume_unit,'گرم')
        self.assertEqual(d.value_unit,'هزار تومان')
        self.assertEqual(d.options()['dimensions'][0],'Market')
        self.assertEqual(set(d.meta[:,d.all_dimensions().index('Market')]),
                         {'All','City','Region'})
        with self.assertRaisesRegex(ValueError,'Market'):
            future_request(d,{'metric':'volume_kg','horizon':1})
        forecasts={}
        for level in ('All','City','Region'):
            output=future_request(d,{'metric':'volume_kg','horizon':1,
                                     'filters':{'Market':level}})
            forecasts[level]=output['prediction']
            self.assertEqual(output['target'],'2026-08')
            self.assertEqual(output['periodicity'],'bimonthly')
            self.assertEqual(output['rows'],1)
        self.assertEqual(forecasts,{'All':101.5,'City':21.5,'Region':36.5})
        cached=Dataset.file(path,{'periodicity':'bimonthly'})
        self.assertEqual(cached.periodicity,'bimonthly')
        self.assertEqual(cached.options()['dimensions'],d.options()['dimensions'])

    def test_monthly_workbook_unchanged_and_frequency_mismatch_rejected(self):
        path=self.root/'monthly.xlsx'
        make_xlsx(path,["Apr' 2026","May' 2026","Jun' 2026"])
        d=Dataset.file(path,{'periodicity':'monthly'})
        self.assertEqual(d.months,['2026-04','2026-05','2026-06'])
        forecast=future_request(d,{'metric':'volume_kg','horizon':1,
                                   'filters':{'Market':'City'}})
        self.assertEqual(forecast['target'],'2026-07')
        with self.assertRaises(ValueError):
            Dataset.file(path,{'periodicity':'bimonthly'})

    def test_two_month_seasonal_features_use_six_periods_per_year(self):
        months=[str(p) for p in pd.period_range('2023-02',periods=20,freq='2M')]
        # Construct the exact same end-month cadence as the bimonthly parser.
        meta=np.array([['Tehran','Co','A','Main','Water','Still','X','X','City']])
        numbers=np.arange(1,21,dtype=float)[None,:]
        d=Dataset(meta,numbers,numbers*10,months,'synthetic','گرم','هزار تومان',
                  app_core.DIMS+['Market'],None,'bimonthly')
        f=Features(d,'volume_kg',0,19)
        features,_,_=f.features(5,1)
        self.assertEqual(features['target_month'].iloc[0],2)
        self.assertEqual(features['target_month_previous_year'].iloc[0],1)
        self.assertIn('Market',features.columns)


if __name__=='__main__':
    unittest.main()
