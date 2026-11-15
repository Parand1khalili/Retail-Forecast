"""Verify dependency diagnosis catches Python mismatches before expensive forecasts."""
import sys
import unittest
from types import SimpleNamespace

from runtime_preflight import diagnose, require_ready, install_command


class RuntimePreflightTests(unittest.TestCase):
    def test_correct_interpreter_with_catboost_reports_ready(self):
        def load(name):
            return SimpleNamespace(__version__='1.2.10',
                                   CatBoostRegressor=lambda: None)
        report = diagnose(load, executable='C:\\Python314\\python.exe',
                          requirements='C:\\My Project\\forecasting\\requirements.txt')
        self.assertTrue(report['ready'])
        self.assertEqual(report['packages']['catboost'],'1.2.10')
        self.assertEqual(report['failures'],[])
        self.assertIn('"C:\\Python314\\python.exe" -m pip install',report['install_command'])
        require_ready(report)

    def test_missing_catboost_for_active_python_is_reported(self):
        def load(name):
            if name=='catboost':
                raise ModuleNotFoundError("No module named 'catboost'")
            return SimpleNamespace(__version__='2.5.3')
        report = diagnose(load, executable='C:\\Venv With Spaces\\Scripts\\python.exe')
        self.assertFalse(report['ready'])
        self.assertEqual(report['failures'][0]['package'],'catboost')
        self.assertIn('No module named',report['failures'][0]['reason'])
        self.assertIn('C:\\Venv With Spaces\\Scripts\\python.exe',report['install_command'])
        self.assertIn('requirements.txt',report['install_command'])
        with self.assertRaisesRegex(RuntimeError,'CatBoost|catboost'):
            require_ready(report)

    def test_dll_or_abi_failure_is_not_mistaken_for_installed_package(self):
        def load(name):
            if name == 'catboost':
                raise OSError('[WinError 126] DLL load failed')
            return SimpleNamespace(__version__='2')
        report = diagnose(load)
        self.assertFalse(report['ready'])
        self.assertEqual(report['failures'][0]['error_type'],'OSError')
        self.assertIn('DLL',report['failures'][0]['reason'])

    def test_broken_catboost_api_is_not_ready(self):
        result = diagnose(lambda name:SimpleNamespace(__version__='1.2.10'))
        self.assertFalse(result['ready'])
        self.assertIn('CatBoostRegressor',result['failures'][0]['reason'])

    def test_cli_is_ascii_json_under_windows_legacy_codepage(self):
        import json
        import os
        from pathlib import Path
        import subprocess
        env = dict(os.environ, PYTHONIOENCODING='cp1252:strict')
        process = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('runtime_preflight.py'))],
            text=True, capture_output=True, env=env, timeout=45)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(process.stdout.isascii())
        output = json.loads(process.stdout)
        self.assertTrue(output['ready'])
        self.assertIn('catboost', output['packages'])
        self.assertEqual(output['python_executable'], sys.executable)

    def test_uses_interpreter_not_a_global_pip_binary(self):
        cmd=install_command(executable='C:\\Python314\\python.exe',
                            requirements='H:\\Repo (backup)\\forecasting\\requirements.txt')
        self.assertIn('"C:\\Python314\\python.exe" -m pip install',cmd)
        self.assertIn('"H:\\Repo (backup)\\forecasting\\requirements.txt"',cmd)


if __name__ == '__main__':
    unittest.main()
