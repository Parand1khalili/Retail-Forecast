"""Source identity captured at startup, not recomputed by a stale server."""
import hashlib
import json
from pathlib import Path
import sys


def identity():
    folder = Path(__file__).resolve().parent
    paths = {name: str(folder / name) for name in (
        'app.py', 'app_core.py', 'schema.py', 'auto_engine.py', 'backtest.py',
        'light_benchmark.py', 'runtime_identity.py', 'app_ui.html')}
    digest = hashlib.sha256()
    for name, path in paths.items():
        digest.update(name.encode())
        digest.update(Path(path).read_bytes())
    return {'root': str(folder.parent), 'python': sys.executable,
            'source_id': digest.hexdigest(), 'paths': paths}


if __name__ == '__main__':
    print(json.dumps(identity()))
