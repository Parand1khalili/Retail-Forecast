"""Deterministic microbenchmark for selection/registry overhead (no CatBoost fit).

Run: python forecasting/benchmark_selection_overhead.py
Measures metadata validation and the Champion–Challenger decision only.
Never interpret these numbers as full-model forecasting runtime.
"""
import json
import time
import tracemalloc

import numpy as np
from selection_core import eligible_candidates, choose_champion


def main():
    rng = np.random.default_rng(42)
    origins = (23, 26, 29)
    series = 3000
    names = (
        'base|Last month|2', 'base|Mean 2|2', 'base|Mean 3|3',
        'base|Mean 6|6', 'base|Seasonal naive|12', 'base|Damped trend|6',
        'ml|raw|24', 'ml|scaled|24', 'ml|log|24',
        'ml|raw|all', 'ml|scaled|all', 'ml|log|all'
    )
    tracemalloc.start()
    start = time.perf_counter()
    actual = {origin: np.maximum(0., rng.normal(100, 25, (series, 3)))
              for origin in origins}
    grid = {origin: {
        name: np.maximum(0., actual[origin] + rng.normal(0, 12 + i, (series, 3)))
        for i, name in enumerate(names)}
        for origin in origins}
    accepted, rejected = eligible_candidates(
        grid, origins, 'volume_kg', 'monthly', series)
    losses = {name: np.array([
        np.abs(grid[origin][name][:, 0] - actual[origin][:, 0]).sum()
        for origin in origins]) for name in accepted}
    winner, decision = choose_champion(losses, 'base|Last month|2')
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(json.dumps({
        'benchmark': 'selection-only', 'seed': 42, 'series': series,
        'candidate_count': len(names), 'origins': len(origins),
        'accepted': len(accepted), 'rejected': len(rejected),
        'winner': winner, 'legacy_best': decision['legacy_best'],
        'seconds': round(elapsed, 4), 'peak_python_mib': round(peak / (1024**2), 3),
        'excluded_from_measurement': 'CatBoost fitting, Excel ingestion, feature engineering, UI, export'
    }))
    if len(accepted) != len(names) or rejected:
        raise RuntimeError('Candidate microbenchmark unexpectedly rejected a valid method.')


if __name__ == '__main__':
    main()
