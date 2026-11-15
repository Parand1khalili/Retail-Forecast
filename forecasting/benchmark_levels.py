"""Disjoint, per-level out-of-sample comparison for forecast Recipes.

Report each level independently. Never sum Market/Category/Brand/SKU summaries
into a second total; those rows describe overlapping views of the same sales.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


PLACEHOLDER = {'', 'all', 'not specified', 'nan', 'none', '—', '-'}
CATEGORY = ('Subcategory', 'Subcategory2', 'Subcategory3', 'Subcategory4')


def error_metrics(actual, predicted):
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Benchmark needs identical finite actual and forecast cells.')
    diff = p - a
    total = float(a.sum())
    absolute = float(np.abs(diff).sum())
    return {'actual': total, 'predicted': float(p.sum()),
            'absolute_error': absolute,
            'wape': absolute / total if total > 0 else None,
            'mae': float(np.abs(diff).mean()) if len(a) else None,
            'bias': float(diff.sum() / total) if total > 0 else None,
            'forecast_actual_ratio': float(p.sum() / total) if total > 0 else None,
            'zero_actual_cells': int((a == 0).sum()),
            'cells': len(a)}


def _levels(dataset, origin, include_series=True):
    dimensions = dataset.all_dimensions()
    meta = dataset.meta
    n = len(meta)
    yield 'Market', {'All selected series': np.arange(n, dtype=int)}
    categories = [d for d in CATEGORY if d in dimensions and
                  any(str(v).strip().casefold() not in PLACEHOLDER for v in meta[:, dimensions.index(d)])]
    for depth in range(1, min(3, len(categories)) + 1):
        cols = [dimensions.index(d) for d in categories[:depth]]
        groups = defaultdict(list)
        for i, row in enumerate(meta):
            key = tuple(str(row[c]).strip() for c in cols)
            if all(v.casefold() not in PLACEHOLDER for v in key):
                groups[' / '.join(key)].append(i)
        if groups:
            yield categories[depth - 1], {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    if 'Brand' in dimensions:
        groups = defaultdict(list)
        for i, row in enumerate(meta):
            name = str(row[dimensions.index('Brand')]).strip()
            groups[name].append(i)
        yield 'Brand', {k: np.asarray(v, dtype=int) for k,v in groups.items()}
    # Each series is an observed product/location cell, NOT an inferred SKU ID.
    if include_series:
        yield 'Series', {str(i): np.array([i]) for i in range(n)}
    recent = dataset.metric('volume_kg')[:, max(0, origin - 11):origin + 1]
    positive = (recent > 0).sum(axis=1)
    masks = [('Zero recent volume', positive == 0),
             ('Intermittent (1–2 active periods)', (positive > 0) & (positive <= 2)),
             ('Regular (3+ active periods)', positive >= 3)]
    sparse = {name: np.flatnonzero(mask) for name, mask in masks if mask.any()}
    if sparse:
        yield 'Demand pattern', sparse


def compare_levels(dataset, actual, reference, challenger, origin, legacy=None,
                   include_series=True):
    """Per-group reporting; no holdout labels are used to form groups."""
    actual = np.asarray(actual, float)
    reference = np.asarray(reference, float)
    challenger = np.asarray(challenger, float)
    legacy = reference if legacy is None else np.asarray(legacy, float)
    if any(x.shape != actual.shape for x in (reference, challenger, legacy)):
        raise ValueError('All benchmark forecasts must cover identical series.')
    reports = []
    for level, groups in _levels(dataset, origin, include_series):
        for name, rows in groups.items():
            if not len(rows):
                continue
            observed = actual[rows]
            ref = error_metrics(observed, reference[rows])
            proposed = error_metrics(observed, challenger[rows])
            previous = error_metrics(observed, legacy[rows])
            baseline_error = ref['absolute_error']
            reports.append({
                'level': level, 'group': name, 'series': len(rows),
                'actual': proposed['actual'], 'global_predicted': ref['predicted'],
                'hierarchical_predicted': proposed['predicted'],
                'global_wape': ref['wape'], 'hierarchical_wape': proposed['wape'],
                'legacy_global_wape': previous['wape'],
                'legacy_global_predicted': previous['predicted'],
                'legacy_global_mae': previous['mae'],
                'legacy_global_bias': previous['bias'],
                'global_mae': ref['mae'], 'hierarchical_mae': proposed['mae'],
                'global_bias': ref['bias'], 'hierarchical_bias': proposed['bias'],
                'global_absolute_error': baseline_error,
                'hierarchical_absolute_error': proposed['absolute_error'],
                'legacy_absolute_error': previous['absolute_error'],
                'relative_reduction_vs_legacy': (
                    (previous['absolute_error']-proposed['absolute_error']) /
                    previous['absolute_error'] if previous['absolute_error']>0 else None),
                'relative_error_reduction': ((baseline_error - proposed['absolute_error']) / baseline_error
                                             if baseline_error > 0 else None),
                'aggregated_global_wape': (abs(ref['predicted'] - ref['actual']) / ref['actual']
                                           if ref['actual'] > 0 else None),
                'aggregated_hierarchical_wape': (abs(proposed['predicted'] - proposed['actual']) / proposed['actual']
                                                 if proposed['actual'] > 0 else None),
                'zero_actual_cells': proposed['zero_actual_cells'],
                'improvement_vs_legacy': (
                    'improved' if proposed['absolute_error'] < previous['absolute_error']-1e-9
                    else 'deteriorated' if proposed['absolute_error'] > previous['absolute_error']+1e-9
                    else 'unchanged'),
                'improvement': ('improved' if proposed['absolute_error'] < baseline_error - 1e-9 else
                                'deteriorated' if proposed['absolute_error'] > baseline_error + 1e-9 else 'unchanged')
            })
    return reports


def summarize_levels(rows):
    """Compact diagnostic overview for UI/cache; full rows belong in audit CSV."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['level']].append(row)
    return {level: {
        'groups': len(group),
        'improved': sum(r['improvement'] == 'improved' for r in group),
        'deteriorated': sum(r['improvement'] == 'deteriorated' for r in group),
        'unchanged': sum(r['improvement'] == 'unchanged' for r in group),
        'undefined_wape_groups': sum(r['hierarchical_wape'] is None for r in group),
        'improved_vs_legacy': sum(r['improvement_vs_legacy']=='improved' for r in group),
        'deteriorated_vs_legacy': sum(r['improvement_vs_legacy']=='deteriorated' for r in group),
        'unchanged_vs_legacy': sum(r['improvement_vs_legacy']=='unchanged' for r in group),
        'group_aggregate_absolute_error': sum(
            abs(r['hierarchical_predicted'] - r['actual']) for r in group),
        'series_level_absolute_error': sum(r['hierarchical_absolute_error'] for r in group),
    } for level, group in groups.items()}
