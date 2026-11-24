"""Conditional unit-value shock competition, separate from ordinary selection.

Selection uses only completed, non-overlapping historical shock episodes.
A unit-value shock is a market-wide price/mix signal, not a causal price effect.
The regular policy is the incumbent, not an automatic shock override.
"""
from copy import deepcopy

import numpy as np

from selection_core import (ChampionGuard, candidate_info, eligible_candidates,
                            choose_champion, candidate_registry)
from category_selection import CategoryGuard, _groups, _evaluate, _stats

GUARD_ID = 'price-guard|robust'
MIN_EVENTS = 2
MAX_EVENTS = 6
HORIZONS = (1, 2, 3)
MIN_ORIGIN = 17


def direction(signal):
    if not signal.get('active'):
        return 0
    return 1 if signal['current_change'] > signal['median_prior_change'] else -1


def shock_origins(cutoff, detect):
    """Use one issuance date per consecutive shock episode, same direction.

    Historical targets H1..H3 must all be observed at issuance. Exclude
    the entire current episode; greedy recent-first spacing prevents reuse
    of any target across events. Detection itself receives only each origin.
    """
    current = direction(detect(cutoff))
    if not current:
        return []
    episodes = []
    previous = 0
    current_start = cutoff
    for origin in range(MIN_ORIGIN, cutoff + 1):
        sign = direction(detect(origin))
        if sign and sign != previous:
            episodes.append((origin, sign))
            if sign == current:
                current_start = origin
        previous = sign
    valid = [o for o, sign in episodes
             if sign == current and o + 3 <= cutoff and o + 3 < current_start]
    selected = []
    for origin in reversed(valid):
        if not selected or selected[-1] - origin >= 3:
            selected.append(origin)
        if len(selected) == MAX_EVENTS:
            break
    return sorted(selected)


def _catalog(entry, common):
    """Use existing ordinary blends unchanged, avoiding ambiguous blend IDs."""
    catalog = {name: [[name, 1.]] for name in sorted(common)}
    for item in entry.get('selection_evidence', {}).get('candidate_scores', []):
        if all(name in common for name, _ in item['components']):
            catalog[item['name']] = item['components']
    return catalog


def _compare(dataset, origins, horizon, rows, actual, grid, catalog, incumbent,
             category=False):
    reference = incumbent['name']
    parts = incumbent['components']
    if any(name not in grid[o] for o in origins for name, _ in parts):
        return incumbent, None
    common = set(catalog)
    if any(name not in {part for recipe in catalog.values() for part, _ in recipe}
           for name, _ in parts):
        return incumbent, None
    recipes = dict(catalog)
    # Keep the exact incumbent composition if a name was already in catalog.
    recipes[reference] = parts
    # Avoid retaining an N-series × N-events matrix for every candidate.
    # Large RA workbooks can have hundreds of thousands of cells.
    losses = {}
    score_rows = []
    for name, pieces in recipes.items():
        pred = _evaluate(pieces, grid, origins, horizon, rows)
        losses[name] = np.abs(pred - actual).sum(axis=0)
        if not category:
            score = _stats(actual, pred)
            score_rows.append({'name': name, 'components': pieces,
                               'wape': score['wape'], 'bias': score['bias'],
                               'absolute_error': float(losses[name].sum())})
    if category:
        guard = CategoryGuard()
        ref = losses[reference]
        total = float(ref.sum())
        eligible = [name for name in common if name != reference and total > 0
                    and (total - losses[name].sum()) / total >= guard.min_relative_improvement
                    and np.all(losses[name] < ref)]
        winner = min(eligible, key=lambda name: (float(losses[name].sum()), name)) if eligible else reference
        decision = {'selected': winner, 'reference': reference,
                    'reason': 'stable category improvement' if eligible else 'category fallback'}
    else:
        winner, decision = choose_champion(losses, reference, ChampionGuard())
    predicted = _evaluate(recipes[winner], grid, origins, horizon, rows)
    reference_prediction = _evaluate(parts, grid, origins, horizon, rows)
    score_rows.sort(key=lambda row: (row['absolute_error'], row['name']))
    evidence = {'origins': len(origins), 'reference': reference,
                'selected': winner, 'decision': decision,
                'group': _stats(actual, predicted),
                'inherited': _stats(actual, reference_prediction),
                'relative_improvement': (
                    float(1 - losses[winner].sum()/losses[reference].sum())
                    if losses[reference].sum() > 0 else None),
                'per_origin_absolute_error': losses[winner].tolist()}
    if not category:
        evidence['candidate_scores'] = score_rows
        evidence['candidate_registry'] = candidate_registry(recipes)
        evidence['origin_details'] = [
            {'origin': dataset.months[o], 'target': dataset.months[o+horizon],
             'actual': float(actual[:, i].sum()),
             'predicted': float(predicted[:, i].sum()),
             'wape': _stats(actual[:, i], predicted[:, i])['wape'],
             'absolute_error': float(losses[winner][i])}
            for i, o in enumerate(origins)]
    return {'name': winner, 'components': recipes[winner]}, evidence


def _category_eligible(actual):
    guard = CategoryGuard()
    sales = actual.sum(axis=1)
    total = float(sales.sum())
    return (len(sales) >= guard.min_series and
            int((sales > 0).sum()) >= guard.min_positive_series and total > 0 and
            total**2 / np.dot(sales, sales) >= guard.min_effective_series and
            sales.max()/total <= guard.max_single_series_share and
            np.all(actual.sum(axis=0) > 0))


def conditional_policy(dataset, cutoff, normal, origins, grid, live_names):
    """Return a copy of the ordinary policy, with evidence-gated overrides.

    All candidates must be available at every matched origin and live.
    Category groups retain their ordinary override as incumbent when it exists;
    otherwise they inherit the effective parent. No group is fitted separately.
    """
    result = deepcopy(normal)
    audit = {'active': True, 'origins': [dataset.months[o] for o in origins],
             'minimum_events': MIN_EVENTS, 'horizons': {}, 'reason': 'insufficient_events'}
    result['_price_regime'] = audit
    if len(origins) < MIN_EVENTS:
        return result
    if (origins != sorted(set(origins)) or any(o < MIN_ORIGIN or o + 3 > cutoff for o in origins)
            or any(b-a < 3 for a,b in zip(origins,origins[1:]))):
        raise ValueError('Invalid or overlapping shock validation origins.')
    common, rejected = eligible_candidates(grid, origins, 'value_1000_toman',
                                           dataset.periodicity, len(dataset.meta))
    common &= set(live_names)
    audit['rejected_candidates'] = rejected
    audit['candidate_registry'] = candidate_registry(common)
    if GUARD_ID not in common:
        audit['reason'] = 'guard_unavailable'
        return result
    audit['reason'] = 'matched_event_competition'
    hierarchy = result['_hierarchy']
    columns = hierarchy['dimensions']
    groups = (_groups(dataset, columns) if columns and normal.get('_strategy')!='global' else {})
    normal_hierarchy = normal['_hierarchy']
    all_rows = np.arange(len(dataset.meta))
    for h in HORIZONS:
        key = str(h)
        entry = normal[key]
        catalog = _catalog(entry, common)
        actual = np.column_stack([dataset.value[:, o+h] for o in origins])
        selected, evidence = _compare(dataset, origins, h, all_rows, actual, grid,
                                      catalog, entry)
        if evidence is None:
            audit['horizons'][key] = {'reason': 'incumbent_unavailable_on_matched_origins'}
            continue
        audit['horizons'][key] = evidence
        result[key] = {**entry, **selected,
                       'basis': 'رقابت مدل‌ها در رخدادهای هم‌جهت ارزش واحد؛ شواهد تاریخی مشروط',
                       'validation_wape': evidence['group']['wape'],
                       'validation_origins': audit['origins'],
                       'selection_decision': evidence['decision'],
                       'selection_evidence': {
                           'metric': 'pooled WAPE', 'grain': entry.get('comparison_grain','series'),
                           'scope': 'matched_shock_events', 'horizon': h,
                           'selected': selected['name'],
                           'candidate_scores': evidence['candidate_scores'],
                           'candidate_registry': evidence['candidate_registry'],
                           'selection_decision': evidence['decision'],
                           'origin_details': evidence['origin_details'],
                           'rejected_candidates': rejected}}
        ordinary = {tuple(node['path']): node
                    for node in normal_hierarchy['overrides'].get(key, [])}
        chosen = {}
        overrides = []
        for path, rows in sorted(groups.items(), key=lambda pair: (len(pair[0]), pair[0])):
            parent = chosen.get(path[:-1], result[key])
            incumbent = ordinary.get(path, parent)
            winner, ev = incumbent, None
            if _category_eligible(actual[rows]):
                winner, ev = _compare(dataset, origins, h, rows, actual[rows], grid,
                                      catalog, incumbent, category=True)
            chosen[path] = winner
            if winner['components'] != parent['components']:
                if ev is None or winner['components'] == incumbent['components']:
                    node = dict(ordinary.get(path, winner))
                    # Retained ordinary evidence must not be called shock evidence.
                    node.update(path=list(path), depth=len(path),
                                parent_recipe=parent['name'])
                else:
                    node = {**winner, 'path': list(path), 'depth': len(path),
                            'parent_recipe': parent['name'], 'evidence': ev,
                            'selection_scope': 'matched_shock_events'}
                overrides.append(node)
        hierarchy['overrides'][key] = overrides
    hierarchy['groups_considered'] = len(groups)*len(HORIZONS)
    hierarchy['groups_selected'] = sum(len(v) for v in hierarchy['overrides'].values())
    return result
