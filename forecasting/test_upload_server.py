"""HTTP upload and Windows-cleanup regression tests; no private Excel data required."""
import http.client
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app_core


class UploadServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(app_core, "ROOT", Path(self.temp.name)), patch.object(
            app_core.Dataset, "builtin", side_effect=FileNotFoundError
        ):
            spec = importlib.util.spec_from_file_location(
                "forecasting_upload_test_app", Path(__file__).with_name("app.py")
            )
            self.app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.app)
        self.server = self.app.ThreadingHTTPServer(("127.0.0.1", 0), self.app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, route, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, route, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_full_export_download_and_missing_dataset(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request('GET', '/api/export/full')
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn('error', json.loads(response.read()))
        finally:
            connection.close()
        import full_export
        self.app.state['dataset'] = object()
        def fake_export(dataset, destination, strategy='hierarchical'):
            self.assertEqual(strategy,'hierarchical')
            self.assertIs(dataset, self.app.state['dataset'])
            destination.write_bytes(b'test-xlsx-bytes')
        with patch.object(full_export, 'export_workbook', side_effect=fake_export):
            connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
            try:
                connection.request('GET', '/api/export/full')
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn('spreadsheetml.sheet', response.getheader('Content-Type'))
                self.assertIn('attachment', response.getheader('Content-Disposition'))
                self.assertEqual(response.read(), b'test-xlsx-bytes')
            finally:
                connection.close()

    def test_full_export_strategy_query_validation(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request('GET', '/api/export/full?strategy=invalid')
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn('error', json.loads(response.read()))
        finally:
            connection.close()

    def test_model_lab_http_run_download_and_upload_invalidation(self):
        from io import BytesIO
        from zipfile import ZipFile
        import backtest_lab

        status, response = self.request('GET', '/api/backtest/report')
        self.assertEqual(status, 400)
        self.assertIn('error', response)
        status, response = self.request(
            'POST', '/api/backtest',
            body=json.dumps({'market':'All','metrics':['volume_kg']}).encode())
        self.assertEqual(status, 400)

        class Imported:
            source = 'test_master.xlsx'
            def for_market(self, market):
                if market != 'All':
                    raise ValueError('Unselected Market')
                return self
            def for_grain(self, grain):
                return self
            def options(self):
                return {'source': self.source, 'months': ['2026-05','2026-06']}
        dataset = Imported()
        self.app.state['dataset'] = dataset
        buffer = BytesIO()
        with ZipFile(buffer, 'w') as archive:
            archive.writestr('volume_kg_summary.csv', 'metric,wape\nvolume_kg,0.1')
        contents = buffer.getvalue()
        expected = {'market':'All', 'metrics':['volume_kg'], 'summaries':[]}
        with patch.object(backtest_lab, 'run_backtest',
                          return_value=(expected, contents)) as run:
            status, payload = self.request(
                'POST', '/api/backtest',
                body=json.dumps({'market':'All','metrics':['volume_kg']}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(payload, expected)
        run.assert_called_once_with(dataset, ['volume_kg'])

        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request('GET', '/api/backtest/report')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Content-Type'), 'application/zip')
            self.assertIn('attachment', response.getheader('Content-Disposition'))
            self.assertEqual(response.read(), contents)
        finally:
            connection.close()
        # Replacing the workbook removes access to the previous workbook report.
        with patch.object(self.app.Dataset, 'file', return_value=Imported()):
            status, _ = self.request(
                'POST', '/api/upload', body=b'synthetic-xlsx-body',
                headers={'X-Filename':'other.xlsx',
                         'X-Periodicity':'monthly'})
        self.assertEqual(status, 200)
        self.assertIsNone(self.app.state['benchmark_zip'])
        self.assertEqual(self.request('GET', '/api/backtest/report')[0], 400)

    def test_model_lab_rejects_invalid_metric_without_running(self):
        import backtest_lab
        class DatasetStub:
            def for_market(self, market): return self
            def for_grain(self, grain): return self
        self.app.state['dataset'] = DatasetStub()
        with patch.object(backtest_lab, 'run_backtest',
                          side_effect=ValueError('شاخص فروش نامعتبر است.')) as run:
            status, body = self.request('POST', '/api/backtest',
                                        json.dumps({'metrics':['invalid']}).encode())
        self.assertEqual(status, 400)
        self.assertIn('error', body)
        self.assertEqual(run.call_count, 1)

    def test_no_sample_still_allows_uploading(self):
        status, body = self.request("GET", "/api/data")
        self.assertEqual(status, 200)
        self.assertFalse(body["ready"])

    def test_binary_upload_updates_dataset_and_removes_temporary_file(self):
        class Imported:
            source = ""

            def options(self):
                return {"source": self.source, "months": ["2026-05", "2026-06"]}

        seen = []

        def parse(path, mapping=None):
            seen.append((path, path.read_bytes(), mapping))
            return Imported()

        with patch.object(self.app.Dataset, "file", side_effect=parse):
            status, result = self.request(
                "POST", "/api/upload", body=b"synthetic-xlsx-body",
                headers={"Content-Length": "19", "X-Filename": "Water%20June.xlsx",
                         "X-Periodicity": "bimonthly"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(result["source"], "Water June.xlsx")
        self.assertEqual(seen[0][1], b"synthetic-xlsx-body")
        self.assertEqual(seen[0][2]["periodicity"], "bimonthly")
        self.assertFalse(seen[0][0].exists())

    def test_cleanup_retries_windows_file_lock(self):
        class FlakyFile:
            attempts = 0

            def unlink(self, missing_ok=False):
                self.attempts += 1
                if self.attempts < 3:
                    raise PermissionError("file locked")

        path = FlakyFile()
        with patch.object(self.app.time, "sleep"):
            self.app.cleanup_upload(path)
        self.assertEqual(path.attempts, 3)

    def test_local_path_accepts_surrounding_quotes(self):
        path = Path(self.temp.name) / "master.xlsx"
        path.write_bytes(b"test")

        class Imported:
            source = "master.xlsx"

            def options(self):
                return {"source": self.source, "months": ["2026-05", "2026-06"]}

        with patch.object(self.app.Dataset, "file", return_value=Imported()) as parse:
            body = json.dumps({"path": f'  "{path}"  ', "periodicity": "bimonthly"}).encode()
            status, response = self.request(
                "POST", "/api/load", body=body,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(response["source"], "master.xlsx")
        parse.assert_called_once()
        self.assertEqual(parse.call_args.args[1]['periodicity'], 'bimonthly')


if __name__ == "__main__":
    unittest.main()
