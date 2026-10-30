"""Numerical invariants for the lightweight benchmark, independent of source data."""
import numpy as np
from light_benchmark import candidates

v = np.array([[100.] * 24, [0.] * 24])
for metric in ['volume_kg', 'value_1000_toman']:
    for h in [1, 2, 3]:
        predictions = candidates(v, v*10, metric, h)
        for name, p in predictions.items():
            assert np.isfinite(p).all() and (p >= 0).all(), name
            assert p[1] == 0, name
            if name != 'croston_sba':
                assert np.isclose(p[0], 100 if metric == 'volume_kg' else 1000), name
        assert np.isclose(predictions['croston_sba'][0], 95 if metric == 'volume_kg' else 950)
# Seasonal forecast must use the matching month, not the last calendar month.
ramp = np.arange(1., 25.)[None, :]
for h in [1, 2, 3]:
    assert candidates(ramp, ramp*10, 'volume_kg', h)['seasonal12'][0] == 12+h
print('PASS: nonnegative, constant, all-zero, SBA correction, seasonal indexing')
