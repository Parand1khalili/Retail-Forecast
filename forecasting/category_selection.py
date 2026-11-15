"""Conservative category-aware selection above globally trained forecasting candidates.

A candidate is trained once per Market/metric/origin; only its per-series outputs
are regrouped here. This module never fits a per-category ML model. It also
never uses the independent holdout to tune or select recipes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from benchmark_levels import compare_levels, summarize_levels


@dataclass(frozen=True)
class CategoryGuard:
    # Predeclared thresholds. Never tune them using the independent holdout.
    min_origins: int = 2
    min_series: int = 6
    min_positive_series: int = 4
    min_effective_series: float = 3.0
    min_relative_improvement: float = 0.05
    max_single_series_share: float = 0.65


PLACEHOLDERS = {'', 'all', 'not specified', 'not available', 'none', 'nan', '-', '—', 'نامشخص'}
CATEGORY_FIELDS = ('Subcategory', 'Subcategory2', 'Subcategory3', 'Subcategory4')


def category_dimensions(dataset):
    """Use mapped category fields only, never arbitrary brand, SKU or geography."""
    dims = dataset.all_dimensions()
    cols = []
    for name in CATEGORY_FIELDS:
        if name in dims:
            index = dims.index(name)
            if any(str(v).strip().casefold() not in PLACEHOLDERS for v in dataset.meta[:, index]):
                cols.append(name)
        if len(cols) == 3:
            break
    return cols


def _groups(dataset, columns):
    dims = dataset.all_dimensions()
    indices = [dims.index(c) for c in columns]
    grouped = {}
    for index, row in enumerate(dataset.meta):
        path = []
        for col in indices:
            value = str(row[col]).strip()
            if value.casefold() in PLACEHOLDERS:
                break
            path.append(value)
            grouped.setdefault(tuple(path), []).append(index)
    return {key: np.asarray(indexes, dtype=int) for key, indexes in grouped.items()}


def _stats(actual, predicted):
    error = predicted - actual
    total = float(actual.sum())
    absolute = float(np.abs(error).sum())
    return {'wape': absolute / total if total > 0 else None,
            'mae': float(np.abs(error).mean()) if error.size else None,
            'bias': float(error.sum() / total) if total > 0 else None,
            'absolute_error': absolute, 'actual': total,
            'predicted': float(predicted.sum()),
            'forecast_actual_ratio': float(predicted.sum() / total) if total > 0 else None,
            'observations': int(actual.size)}


def _evaluate(components, by_origin, origins, horizon, rows):
    values = []
    for origin in origins:
        matrix = sum(weight * by_origin[origin][name][rows, horizon - 1]
                     for name, weight in components)
        values.append(np.asarray(matrix, dtype=float))
    return np.column_stack(values)


def select_category_overrides(dataset, metric, origins, by_origin, global_recipes,
                              guard=CategoryGuard()):
    """Promote category recipes only when every selection origin improves."""
    columns = category_dimensions(dataset)
    policy = {'dimensions': columns, 'guard': asdict(guard),
              'origins': [dataset.months[o] for o in origins],
              'overrides': {}, 'groups_considered': 0, 'groups_selected': 0,
              'selection_note': 'Overrides require positive, stable gains across multiple origins.'}
    if not columns or len(origins) < guard.min_origins:
        return policy
    dims = dataset.all_dimensions()
    if 'Market' in dims and len(set(dataset.meta[:, dims.index('Market')])) > 1:
        raise ValueError('Category selection requires exactly one geographic Market level.')
    groups = _groups(dataset, columns)
    actual_series = dataset.metric(metric)
    for horizon in (1, 2, 3):
        entry = global_recipes[str(horizon)]
        catalog = {item['name']: item['components']
                   for item in entry['selection_evidence']['candidate_scores']}
        chosen = {}
        actual_full = np.column_stack([actual_series[:, o + horizon] for o in origins])
        overrides = []
        for path, rows in sorted(groups.items(), key=lambda pair: (len(pair[0]), pair[0])):
            policy['groups_considered'] += 1
            parent = chosen.get(path[:-1], entry)
            chosen[path] = parent
            values = actual_full[rows]
            volume = values.sum(axis=1)
            total = float(volume.sum())
            positive = volume[volume > 0]
            effective = float(total ** 2 / np.dot(volume, volume)) if total > 0 else 0.
            concentration = float(volume.max() / total) if total > 0 else 1.
            if (len(rows) < guard.min_series or len(positive) < guard.min_positive_series
                    or effective < guard.min_effective_series
                    or concentration > guard.max_single_series_share
                    or any(values[:, index].sum() <= 0 for index in range(len(origins)))):
                continue
            parent_name = parent['name']
            if parent_name not in catalog:
                continue
            parent_prediction = _evaluate(catalog[parent_name], by_origin, origins, horizon, rows)
            parent_losses = np.abs(parent_prediction - values).sum(axis=0)
            parent_total = float(parent_losses.sum())
            if parent_total <= 0:
                continue
            best_name, best_losses, best_prediction = None, None, None
            for name, components in catalog.items():
                if name == parent_name:
                    continue
                candidate = _evaluate(components, by_origin, origins, horizon, rows)
                losses = np.abs(candidate - values).sum(axis=0)
                if best_losses is None or (float(losses.sum()), name) < (float(best_losses.sum()), best_name):
                    best_name, best_losses, best_prediction = name, losses, candidate
            if best_losses is None:
                continue
            gain = (parent_total - float(best_losses.sum())) / parent_total
            if gain < guard.min_relative_improvement or not np.all(best_losses < parent_losses - 1e-9):
                continue
            node = {'path': list(path), 'name': best_name, 'components': catalog[best_name],
                    'parent_recipe': parent_name, 'depth': len(path),
                    'evidence': {'series': len(rows), 'positive_series': len(positive),
                                 'effective_series': round(effective, 2),
                                 'single_series_share': concentration, 'origins': len(origins),
                                 'relative_improvement': gain,
                                 'group': _stats(values, best_prediction),
                                 'inherited': _stats(values, parent_prediction),
                                 'aggregate_wape': _stats(values.sum(axis=0), best_prediction.sum(axis=0))['wape'],
                                 'per_origin_absolute_error': best_losses.tolist()}}
            chosen[path] = node
            overrides.append(node)
            policy['groups_selected'] += 1
        policy['overrides'][str(horizon)] = overrides
    return policy


def series_recipes(dataset, global_recipe, hierarchy, horizon):
    """Assign every existing series the deepest reliable selected recipe."""
    recipes = [global_recipe] * len(dataset.meta)
    columns = hierarchy.get('dimensions', [])
    if not columns:
        return recipes
    groups = _groups(dataset, columns)
    for node in sorted(hierarchy.get('overrides', {}).get(str(horizon), []),
                       key=lambda item: item['depth']):
        for i in groups.get(tuple(node['path']), ()):
            recipes[int(i)] = node
    return recipes


def forecast_with_recipes(dataset, horizon, components, global_recipe, hierarchy,
                          strategy='hierarchical'):
    """Apply cached candidate vectors without training additional group models."""
    recipes = (series_recipes(dataset, global_recipe, hierarchy, horizon)
               if strategy == 'hierarchical' else [global_recipe] * len(dataset.meta))
    output = np.empty(len(dataset.meta), dtype=float)
    usage = {}
    distinct = {recipe['name']: recipe for recipe in recipes}
    for name, recipe in distinct.items():
        rows = np.array([i for i, chosen in enumerate(recipes) if chosen['name'] == name], dtype=int)
        usage[name] = len(rows)
        output[rows] = sum(weight * components[component][rows, horizon - 1]
                           for component, weight in recipe['components'])
    if not np.isfinite(output).all() or (output < 0).any():
        raise ValueError('Category recipe generated invalid or negative predictions.')
    return output, usage, recipes


def holdout_comparison(dataset, metric, holdout_origin, candidates_at_holdout,
                        global_recipes, hierarchy, legacy_recipes=None):
    """Independent audit. This information MUST NOT influence recipe selection."""
    result = {'origin': dataset.months[holdout_origin], 'used_for_selection': False,
              'horizons': {}, 'note': 'One untouched historical origin; not a guarantee of future accuracy.'}
    for horizon in (1, 2, 3):
        if holdout_origin + horizon >= len(dataset.months):
            continue
        global_recipe = global_recipes[str(horizon)]
        legacy_recipe=(legacy_recipes or global_recipes)[str(horizon)]
        def predict(recipe):
            return sum(weight * candidates_at_holdout[name][:, horizon - 1]
                       for name, weight in recipe['components'])
        needed = {name for item in [global_recipe, legacy_recipe] + hierarchy['overrides'].get(str(horizon), [])
                  for name, _ in item['components']}
        if not needed.issubset(candidates_at_holdout):
            continue
        baseline = predict(global_recipe)
        legacy_prediction = predict(legacy_recipe)
        recipes = series_recipes(dataset, global_recipe, hierarchy, horizon)
        distinct = {r['name']: r for r in recipes if r['name'] != global_recipe['name']}
        cache = {name: predict(recipe) for name, recipe in distinct.items()}
        proposed = np.array([cache[recipe['name']][i] if recipe['name'] in cache else baseline[i]
                             for i, recipe in enumerate(recipes)])
        actual = dataset.metric(metric)[:, holdout_origin + horizon]
        global_stats = _stats(actual, baseline)
        hierarchical_stats = _stats(actual, proposed)
        legacy_stats = _stats(actual, legacy_prediction)
        level_rows=compare_levels(dataset,actual,baseline,proposed,holdout_origin,
                                  legacy=legacy_prediction)
        branches = _groups(dataset, hierarchy.get('dimensions', [])[:1]) if hierarchy.get('dimensions') else {}
        improved = deteriorated = unchanged = 0
        for rows in branches.values():
            before = np.abs(baseline[rows] - actual[rows]).sum()
            after = np.abs(proposed[rows] - actual[rows]).sum()
            if after < before - 1e-9: improved += 1
            elif after > before + 1e-9: deteriorated += 1
            else: unchanged += 1
        result['horizons'][str(horizon)] = {
            'groups_improved': improved, 'groups_deteriorated': deteriorated,
            'groups_unchanged': unchanged,
            'global': global_stats, 'hierarchical': hierarchical_stats,
            'legacy_global':legacy_stats,
            'legacy_recipe':legacy_recipe['name'],
            'global_recipe':global_recipe['name'],
            'level_summary':summarize_levels(level_rows),
            'relative_error_reduction_vs_legacy':(
                (legacy_stats['absolute_error'] - hierarchical_stats['absolute_error']) /
                legacy_stats['absolute_error'] if legacy_stats['absolute_error']>0 else None),
            'relative_error_reduction': (
                (global_stats['absolute_error'] - hierarchical_stats['absolute_error']) /
                global_stats['absolute_error'] if global_stats['absolute_error'] > 0 else None),
            'series_using_override': sum(r is not global_recipe for r in recipes)}
    return result


def filtered_holdout_summary(dataset, metric, mask, horizon, holdout_origin,
                             at_origin, global_recipe, hierarchy):
    """Evaluate the current selection only on untouched historical targets.

    A zero actual denominator produces WAPE=None, never a misleading 0%.
    The selection policy is fixed before this function examines holdout actuals.
    """
    if holdout_origin + horizon >= len(dataset.months):
        return None
    required={component for recipe in [global_recipe]+hierarchy['overrides'].get(str(horizon), [])
              for component,_ in recipe['components']}
    if not required.issubset(at_origin):
        return None
    selected=np.asarray(mask,dtype=bool)
    if not selected.any():
        return None
    proposed,_,_=forecast_with_recipes(dataset,horizon,at_origin,global_recipe,hierarchy)
    baseline,_,_=forecast_with_recipes(dataset,horizon,at_origin,global_recipe,hierarchy,'global')
    actual=dataset.metric(metric)[:,holdout_origin+horizon][selected]
    return {'origin':dataset.months[holdout_origin],
            'target':dataset.months[holdout_origin+horizon],
            'series':int(selected.sum()),'independent':True,
            'global':_stats(actual,baseline[selected]),
            'hierarchical':_stats(actual,proposed[selected])}
