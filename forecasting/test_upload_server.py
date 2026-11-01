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
                headers={"Content-Length": "19", "X-Filename": "Water%20June.xlsx"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(result["source"], "Water June.xlsx")
        self.assertEqual(seen[0][1], b"synthetic-xlsx-body")
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
            body = json.dumps({"path": f'  "{path}"  '}).encode()
            status, response = self.request(
                "POST", "/api/load", body=body,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(response["source"], "master.xlsx")
        parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
