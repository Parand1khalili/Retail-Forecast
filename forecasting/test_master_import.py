"""Real HTTP/parser regression on disposable 36-observation RA masters."""
import json
import numpy as np
from unittest.mock import patch
import test_upload_server as upload_tests
from test_period_market import make_xlsx
from app_core import Dataset
from auto_engine import future_request
from schema import SchemaError


def periods():
    names=['Jan-Feb','Mar-Apr','May-Jun','Jul-Aug','Sep-Oct','Nov-Dec']
    return [f'{names[i%6]} {2020+i//6}' for i in range(3,39)]


class MasterImportTests(upload_tests.UploadServerTests):
    def setUp(self):
        super().setUp()
        from pathlib import Path
        import auto_engine, app_core
        root=Path(self.temp.name)
        for module, attribute in [(auto_engine,'OUT'),(app_core,'CACHE')]:
            replacement=patch.object(module,attribute,root)
            replacement.start();self.addCleanup(replacement.stop)

    def master(self, name='master.xlsx', **kwargs):
        path=__import__('pathlib').Path(self.temp.name)/name
        make_xlsx(path,periods(),brands=('Aqua','Beta'),**kwargs)
        return path

    def upload(self, path, frequency='bimonthly'):
        return self.request('POST','/api/upload',path.read_bytes(),
                            {'X-Filename':path.name,'X-Periodicity':frequency})

    def test_full_master_http_and_cache(self):
        path=self.master(missing={0:'0'})
        code,body=self.upload(path)
        self.assertEqual(code,200,body)
        self.assertEqual(len(body['months']),36)
        self.assertEqual(body['months'][-1],'2026-06')
        self.assertEqual(body['units'],{'volume_kg':'گرم','value_1000_toman':'هزار تومان'})
        d=self.app.state['dataset']
        self.assertEqual(set(d.meta[:,d.all_dimensions().index('Market')]),{'All','City','Region'})
        self.assertEqual(len(d.for_market('City').meta),2)
        self.assertTrue(np.all(d.volume[:,0]==0))
        for h,target in [(1,'2026-08'),(2,'2026-10'),(3,'2026-12')]:
            # Exercise real selection/training on the tiny panel, not a mocked forecast.
            result=future_request(d,{'filters':{'Market':'City','Brand':'Aqua'},'horizon':h})
            self.assertEqual(result['target'],target)
            self.assertEqual(result['rows'],1)
            self.assertTrue(np.isfinite(result['prediction']))
        for _ in range(2):
            cached=Dataset.file(path,{'periodicity':'bimonthly'})
            self.assertEqual(cached.months,d.months)
            self.assertEqual(cached.period_months,2)
        with self.assertRaisesRegex(ValueError,'نوع دوره'):
            Dataset.file(path,{'periodicity':'monthly'})
        code,body=self.request('POST','/api/load',json.dumps({'path':str(path),'periodicity':'bimonthly'}))
        self.assertEqual(code,200,body)
        self.assertEqual(body['periodicity'],'bimonthly')
        previous=self.app.state['dataset']
        code,body=self.upload(path,'monthly')
        self.assertEqual(code,400)
        self.assertIs(self.app.state['dataset'],previous)
        self.assertEqual(self.upload(path)[0],200)

    def test_missing_and_invalid_values(self):
        import zipfile
        path=self.master(missing={0:'-'})
        d=Dataset.file(path,{'periodicity':'bimonthly'})
        self.assertEqual(len(d.months),35)  # Entire unavailable leading period, not zero.
        path=self.master('gap.xlsx',missing={10:'-'})
        with self.assertRaisesRegex(ValueError,'پیوسته'):
            Dataset.file(path,{'periodicity':'bimonthly'})
        path=self.master('invalid.xlsx',missing={0:'corrupt'})
        with self.assertRaisesRegex(ValueError,'نامعتبر'):
            Dataset.file(path,{'periodicity':'bimonthly'})
        path=self.master('partial.xlsx')
        with zipfile.ZipFile(path) as z: entries={n:z.read(n) for n in z.namelist()}
        entries['xl/worksheets/sheet1.xml']=entries['xl/worksheets/sheet1.xml'].replace(b'<t>100</t>',b'<t>-</t>',1)
        with zipfile.ZipFile(path,'w') as z:
            for n,v in entries.items():z.writestr(n,v)
        with self.assertRaisesRegex(ValueError,'خالی'):
            Dataset.file(path,{'periodicity':'bimonthly'})

    def test_health_reports_actual_imports(self):
        code,body=self.request('GET','/api/health')
        self.assertEqual(code,200)
        from pathlib import Path
        self.assertEqual(Path(body['runtime']['paths']['app_core.py']).resolve(),Path(__import__('app_core').__file__).resolve())
        self.assertEqual(len(body['runtime']['source_id']),64)
        from runtime_identity import identity
        with patch.object(Path,'read_bytes',return_value=b'changed source'):
            self.assertNotEqual(identity()['source_id'],body['runtime']['source_id'])
            self.assertEqual(self.request('GET','/api/health')[1]['runtime']['source_id'],body['runtime']['source_id'])
