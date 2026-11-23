"""Real HTTP startup tests: EMRC Forecast must not compete with Juice on 8765.

Uses only a process spawned by this test and a disposable OS-assigned port.
No existing user process is inspected, interrupted, or terminated.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen
from urllib.error import URLError

from app import DEFAULT_PORT, valid_port


APP=Path(__file__).with_name('app.py')


def unused_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1',0))
        return listener.getsockname()[1]


class LocalServerPortTests(unittest.TestCase):
    def test_forecast_has_separate_default_port_from_juice(self):
        self.assertEqual(DEFAULT_PORT,8766)
        for value in ('8766','8876',65535):
            self.assertEqual(valid_port(value),int(value))
        for bad in ('abc',None,'-1',0,'80',65536):
            with self.subTest(value=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    valid_port(bad)

    def test_real_http_server_on_requested_port_serves_forecast_not_juice(self):
        port=unused_port()
        proc=subprocess.Popen(
            [sys.executable,'-u',str(APP),'--port',str(port)],
            cwd=str(APP.parent),stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            health=None
            for _ in range(120):
                if proc.poll() is not None:
                    self.fail('Forecast app exited before HTTP health was available: '+
                              proc.stdout.read()[-3000:])
                try:
                    with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=.5) as res:
                        self.assertEqual(res.status,200)
                        health=json.loads(res.read())
                    break
                except (OSError,URLError,TimeoutError):
                    time.sleep(.1)
            self.assertIsNotNone(health,'Forecast service did not start within 12 seconds.')
            self.assertEqual(health['service'],'ra-forecast')
            self.assertEqual(health['runtime']['port'],port)
            self.assertEqual(Path(health['runtime']['root']),APP.parent.parent)
            self.assertTrue(health['ready'])
            with urlopen(f'http://127.0.0.1:{port}/',timeout=5) as res:
                page=res.read().decode('utf-8')
                self.assertEqual(res.status,200)
            self.assertIn('EMRC Retail Forecast | Model Lab',page)
            self.assertIn('اجرای آزمون و مقایسه مدل‌ها',page)
            self.assertNotIn('Juice Forecast Intelligence',page)
        finally:
            # Terminate ONLY our own child, never the dashboard or another server.
            if proc.poll() is None:
                proc.terminate()
                try: proc.communicate(timeout=7)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate(timeout=7)
            else:
                proc.communicate(timeout=7)

    def test_occupied_port_fails_safely_and_suggests_alternative(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as owner:
            owner.bind(('127.0.0.1',0))
            owner.listen(1)
            port=owner.getsockname()[1]
            completed=subprocess.run(
                [sys.executable,'-u',str(APP),'--port',str(port)],
                cwd=str(APP.parent),stdin=subprocess.DEVNULL,
                capture_output=True,text=True,timeout=20)
            self.assertNotEqual(completed.returncode,0)
            output=completed.stdout+completed.stderr
            self.assertIn(f'port {port} is already in use',output)
            self.assertIn('python app.py --port 8876',output)


if __name__=='__main__':
    unittest.main()
