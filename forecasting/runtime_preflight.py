"""Interpreter-specific dependency checks for the local EMRC application.

No package is installed here. The check imports distributions with the active
Python executable, instead of guessing from a separate launcher or pip command.
Keep this module importable even when numpy/catboost are absent.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

REQUIRED = ('numpy', 'pandas', 'catboost')


def install_command(executable=None, requirements=None):
    executable = str(executable or sys.executable)
    requirements = str(requirements or Path(__file__).with_name('requirements.txt'))
    prefix = '& ' if os.name == 'nt' else ''
    return f'{prefix}"{executable}" -m pip install -r "{requirements}"'


def diagnose(import_module=None, executable=None, requirements=None):
    """Check actual importability, not merely pip metadata or module discovery."""
    load = import_module or importlib.import_module
    python = str(executable or sys.executable)
    packages = {}
    failures = []
    for name in REQUIRED:
        try:
            module = load(name)
            if name == 'catboost' and not callable(getattr(module, 'CatBoostRegressor', None)):
                raise ImportError('CatBoostRegressor is not available in this installation')
            packages[name] = str(getattr(module, '__version__', 'installed'))
        except Exception as exc:
            packages[name] = None
            failures.append({'package': name, 'reason': str(exc) or type(exc).__name__,
                             'error_type': type(exc).__name__})
    ready = not failures
    return {
        'ready': ready,
        'python_executable': python,
        'python_version': sys.version.split()[0],
        'packages': packages,
        'failures': failures,
        'install_command': install_command(python, requirements),
        'message': ('موتور پیش‌بینی آماده است.'
                    if ready else
                    'پکیج‌های موتور Forecast برای Python فعال نصب نشده‌اند یا بارگذاری نمی‌شوند. '
                    'دستور نصب نمایش‌داده‌شده را برای همین Python اجرا و سپس برنامه را دوباره راه‌اندازی کنید.'),
    }


def require_ready(report):
    if not report['ready']:
        missing = ', '.join(item['package'] for item in report['failures'])
        raise RuntimeError(
            f"موتور Forecast آماده نیست: {missing}. "
            f"Python: {report['python_executable']}. "
            f"نصب در همین محیط: {report['install_command']}"
        )


if __name__ == '__main__':
    state = diagnose()
    print(json.dumps(state, ensure_ascii=False))
    raise SystemExit(0 if state['ready'] else 2)
