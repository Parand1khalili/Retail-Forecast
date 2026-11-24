"""Nested walk-forward evaluation of the actual interactive forecast policies.

Outer targets are never passed to the engine. Each outer cutoff invokes the
production selector (including its inner validation and price regime) anew.
The current master is a retrospective snapshot, not archived as-of vintages.
"""
from collections import defaultdict
from dataclasses import replace
import csv
from pathlib import Path

import numpy as np

from auto_engine import automatic_forecast
from benchmark_levels import _levels

MIN_HISTORY = 18
MAX_OUTER_ORIGINS = 6
HORIZONS = (1, 2, 3)
METHODS = ('baseline', 'global', 'hierarchical')
SCOPE = 'nested_walk_forward'


def outer_origins(periods):
    """Recent, completed, disjoint three-period target blocks."""
    return list(reversed(list(range(periods - 4, MIN_HISTORY - 2, -3))[:MAX_OUTER_ORIGINS]))


def historical_panel(dataset, origin):
    """Only positive-history series are identifiable without archived vintages.

    All-zero prefixes cannot distinguish known inactive from not-yet-listed
    products. Use a conservative observed-sales proxy; score ALL target rows,
    including new products, with zero forecasts for unidentifiable series.
    """
    volume = dataset.volume[:, :origin + 1].copy()
    value = dataset.value[:, :origin + 1].copy()
    known = ((volume > 0) | (value > 0)).any(axis=1)
    prefix = replace(dataset, meta=dataset.meta[known].copy(),
                     volume=volume[known].copy(), value=value[known].copy(),
                     months=list(dataset.months[:origin + 1]), schema=None)
    return prefix, known


def _forecast(prefix, known, metric, horizon, strategy):
    out = np.zeros(len(known), dtype=float)
    if not known.any():
        return out, {'name': 'no_observed_series'}
    pred, recipe = automatic_forecast(prefix, metric, horizon, strategy=strategy)
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (int(known.sum()),) or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Invalid forecast in nested walk-forward replay.')
    out[known] = pred
    return out, recipe


def _matrix(actual, predictions):
    errors = np.column_stack([p - actual for p in predictions])
    return np.column_stack((actual, np.ones(len(actual)), np.abs(errors), errors))


def _ratio(a, b):
    return float(a / b) if b > 0 else None


def _stats(totals):
    actual, cells = totals[:2]
    result = {'actual': float(actual), 'cells': int(cells)}
    for i, method in enumerate(METHODS):
        absolute, signed = totals[2+i], totals[5+i]
        result.update({method+'_wape': _ratio(absolute, actual),
                       method+'_mae': _ratio(absolute, cells),
                       method+'_bias': _ratio(signed, actual),
                       method+'_absolute_error': float(absolute),
                       method+'_predicted': float(actual+signed)})
    for name, index in [('baseline', 2), ('global', 3)]:
        reference, proposed = totals[index], totals[4]
        result['relative_reduction_vs_'+name] = _ratio(reference-proposed, reference)
        result['improvement_vs_'+name] = ('improved' if proposed < reference else
                                           'deteriorated' if proposed > reference else 'unchanged')
    return result


def _write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(dataset, metric, output, detailed_output=None):
    """Pooled outer errors, per-origin audit, and pooled group/series CSV.

    Baseline is last observed period at every cutoff. Global and Hierarchical
    are the existing live strategies, not locked recipes or a controlled
    ablation of category selection. Forecast policy itself is not modified.
    """
    dataset.metric(metric)
    origins = outer_origins(len(dataset.months))
    dims = dataset.all_dimensions()
    market = str(dataset.meta[0, dims.index('Market')]) if 'Market' in dims else 'N/A'
    common = {'metric': metric, 'market': market, 'periodicity': dataset.periodicity,
              'audit_scope': SCOPE}
    summary_path = Path(output)
    fold_path = summary_path.with_name(summary_path.stem.replace('_summary', '')+'_by_origin.csv')
    details_path = Path(detailed_output) if detailed_output is not None else summary_path.with_name(summary_path.stem+'_by_group.csv')
    if not origins:
        rows = [{**common, 'horizon': str(h), 'audit_type': 'no_independent_holdout',
                 'origin': '', 'origin_count': 0, 'outer_origins': '',
                 'baseline_wape': None, 'global_wape': None, 'hierarchical_wape': None,
                 'relative_reduction_vs_baseline': None, 'selection_only': True}
                for h in HORIZONS]
        _write(summary_path, rows)
        return rows

    n = len(dataset.meta)
    series_totals = {h: np.zeros((n, 8), dtype=float) for h in HORIZONS}
    # Group membership can change for the demand-pattern level. Accumulate
    # sufficient statistics, not averages of WAPEs or repeated group rows.
    groups = defaultdict(lambda: [np.zeros(8), 0, 0, np.zeros(3)])
    folds = []
    for origin in origins:
        prefix, known = historical_panel(dataset, origin)
        baseline = np.zeros(n, dtype=float)
        baseline[known] = prefix.metric(metric)[:, -1]
        frozen = {}
        for h in HORIZONS:
            g, gr = _forecast(prefix, known, metric, h, 'global')
            p, pr = _forecast(prefix, known, metric, h, 'hierarchical')
            frozen[h] = (baseline.copy(), g, p, gr, pr)
        # All six production predictions are fixed before reading outer actuals.
        matrices = {}
        for h, (b, g, p, gr, pr) in frozen.items():
            actual = dataset.metric(metric)[:, origin+h]
            matrix = _matrix(actual, (b, g, p))
            matrices[h] = matrix
            series_totals[h] += matrix
            category = pr.get('category_selection') or {}
            regime = pr.get('price_regime') or {}
            folds.append({**common, 'horizon': str(h),
                          'origin': dataset.months[origin], 'target': dataset.months[origin+h],
                          'training_periods': origin+1, 'known_series': int(known.sum()),
                          'unseen_series': int((~known).sum()),
                          'unseen_actual': float(actual[~known].sum()),
                          'global_recipe': gr.get('name'), 'hierarchical_recipe': pr.get('name'),
                          'global_selection_origins': ','.join(gr.get('validation_origins', [])),
                          'hierarchical_selection_origins': ','.join(pr.get('validation_origins', [])),
                          'category_overrides': category.get('groups_selected', 0),
                          'price_regime_reason': regime.get('reason', 'normal'),
                          'price_shock_events': len(regime.get('origins', [])),
                          **_stats(matrix.sum(axis=0))})
        # Grouping uses only historical activity to classify demand patterns.
        for level, members in _levels(dataset, origin, include_series=True, include_individual=False):
            for name, indices in members.items():
                for h in HORIZONS:
                    key = (h, level, name)
                    total = matrices[h][indices].sum(axis=0)
                    acc = groups[key]
                    acc[0] += total
                    acc[1] = max(acc[1], len(indices))
                    acc[2] += 1
                    acc[3] += np.abs(total[5:8])

    labels = [dataset.months[o] for o in origins]
    rows = []
    for h in HORIZONS:
        totals = series_totals[h].sum(axis=0)
        hf = [f for f in folds if f['horizon'] == str(h)]
        rows.append({**common, 'horizon': str(h), 'audit_type': SCOPE,
                     'origin': labels[0] if len(labels)==1 else labels[0]+' … '+labels[-1],
                     'origin_count': len(labels), 'outer_origins': ','.join(labels),
                     'selection_only': False, 'unseen_actual': sum(f['unseen_actual'] for f in hf),
                     **_stats(totals)})

    def group_row(h, level, name, totals, max_series, count, aggregate):
        return {**common, 'horizon': str(h), 'level': level, 'group': name,
                'series': max_series, 'origin_count': count, **_stats(totals),
                **{'aggregated_'+method+'_wape': _ratio(aggregate[i], totals[0])
                   for i, method in enumerate(METHODS)}}

    def detail_rows():
        for (h, level, name), (totals, max_series, count, aggregate) in sorted(groups.items()):
            yield group_row(h, level, name, totals, max_series, count, aggregate)
        for h in HORIZONS:
            for i, totals in enumerate(series_totals[h]):
                yield group_row(h, 'Series', str(i), totals, 1, len(origins), totals[2:5])

    _write(summary_path, rows)
    _write(fold_path, folds)
    # Stream the full per-series CSV rather than building a giant list.
    iterator = iter(detail_rows())
    first = next(iterator)
    from itertools import chain
    _write(details_path, chain([first], iterator), fields=list(first))
    return rows
