"""Full XLSX export regression tests. Run from repo root:
python -m unittest discover -s forecasting -p 'test_*.py' -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from app_core import Dataset
import full_export

NS = {'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def rows_in_sheet(zip_file, sheet_number):
    root = ET.fromstring(zip_file.read(f'xl/worksheets/sheet{sheet_number}.xml'))
    rows = []
    for row in root.findall('.//x:sheetData/x:row', NS):
        cells = []
        for c in row.findall('x:c', NS):
            if c.get('t') == 'inlineStr':
                cells.append(''.join(t.text or '' for t in c.findall('.//x:t', NS)))
            else:
                v = c.find('x:v', NS)
                cells.append(float(v.text) if v is not None else None)
        rows.append(cells)
    return rows


class FullExportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)/'full.xlsx'
        self.dims = ['City','Company','Brand','Market']
        self.data = Dataset(
            np.array([['All','Co','A','All'],
                      ['Tehran','Co','A','City'],
                      ['Shiraz','Co','B','City']],dtype=str),
            np.array([[100.,110.,120.],[10.,11.,12.],[20.,21.,22.]]),
            np.array([[1000.,1100.,1200.],[100.,110.,120.],[200.,210.,220.]]),
            ['2026-02','2026-04','2026-06'], 'demo','گرم','هزار تومان',
            self.dims,None,'bimonthly')

    def test_every_observed_series_and_three_predictions_in_both_sheets(self):
        calls=[]
        def predict(d, metric, horizon):
            calls.append((tuple(sorted(set(d.meta[:,3]))),metric,horizon))
            # Clearly separate per-level predictions from the original history.
            return d.metric(metric)[:,-1] + 100*horizon, {'name':'fake-for-test'}

        with patch.object(full_export,'automatic_forecast',side_effect=predict):
            result=full_export.export_workbook(self.data,self.path)
        self.assertEqual(len(calls),12)  # 2 market levels x 2 metrics x 3 horizons
        self.assertTrue(all(len(levels)==1 for levels,_,_ in calls))
        self.assertEqual(result['rows'],3)
        self.assertEqual(result['future'],['2026-08','2026-10','2026-12'])
        with ZipFile(self.path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(rows_in_sheet(z,1)),4)
            self.assertEqual(len(rows_in_sheet(z,2)),4)
            volumes=rows_in_sheet(z,1)
            values=rows_in_sheet(z,2)
        self.assertEqual(len(volumes[0]),len(self.dims)+3+3)
        self.assertEqual(volumes[0][-3:],[
            'Forecast | Jul-Aug 2026','Forecast | Sep-Oct 2026',
            'Forecast | Nov-Dec 2026'])
        self.assertEqual(volumes[1][-6:],[100.,110.,120.,220.,320.,420.])
        self.assertEqual(volumes[2][-6:],[10.,11.,12.,112.,212.,312.])
        self.assertEqual(values[3][-6:],[200.,210.,220.,320.,420.,520.])
        self.assertEqual(volumes[1][:4],['All','Co','A','All'])
        self.assertEqual(volumes[2][:4],['Tehran','Co','A','City'])

    def test_monthly_period_labels_and_literal_excel_injection_text(self):
        d=Dataset(np.array([['=HYPERLINK("bad")','Co','A','City']],dtype=str),
                  np.array([[2.,3.]]),np.array([[4.,5.]]),
                  ['2026-05','2026-06'],'demo',dimension_names=self.dims)
        with patch.object(full_export,'automatic_forecast',
                          side_effect=lambda d,metric,h:(np.array([float(h)]),{})):
            full_export.export_workbook(d,self.path)
        with ZipFile(self.path) as z:
            xml=z.read('xl/worksheets/sheet1.xml').decode()
            rows=rows_in_sheet(z,1)
        self.assertEqual(rows[0][-3:],['Forecast | 2026-07','Forecast | 2026-08','Forecast | 2026-09'])
        self.assertEqual(rows[1][0],'=HYPERLINK("bad")')
        self.assertNotIn('<f>',xml)

    def test_no_truncated_workbook_when_forecast_fails(self):
        with patch.object(full_export,'automatic_forecast',side_effect=RuntimeError('training failed')):
            with self.assertRaisesRegex(RuntimeError,'training failed'):
                full_export.export_workbook(self.data,self.path)
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
