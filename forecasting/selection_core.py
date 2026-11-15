"""Shared configuration and pure policies for trustworthy forecast competitions.

Candidate forecasts are produced by the existing global engine and are fitted
only through the historical origin. This module does not fit any new models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class BacktestConfig:
    max_selection_origins: int = 3
    selection_stride: int = 3
    min_selection_origin: int = 17  # 18 completed observations for ML eligibility
    min_independent_cutoff: int = 29  # 30 completed observations
    max_horizon: int = 3


@dataclass(frozen=True)
class ChampionGuard:
    min_selection_origins: int = 2
    min_relative_improvement: float = 0.02
    min_winning_origin_fraction: float = 2 / 3
    max_relative_origin_worsening: float = 0.05


@dataclass(frozen=True)
class CandidateInfo:
    name: str
    family: str
    display_name: str
    metrics: tuple[str, ...]
    periodicities: tuple[str, ...]
    min_history: int
    resource: str
    cache: str
    deterministic: bool = True

    def record(self):
        item = asdict(self)
        item['metrics'] = list(self.metrics)
        item['periodicities'] = list(self.periodicities)
        return item


_METRICS = ('volume_kg', 'value_1000_toman')
_FREQUENCIES = ('monthly', 'bimonthly')


def candidate_info(name):
    """Describe actual candidate IDs without changing existing model training."""
    parts = name.split('|')
    family = parts[0]
    metrics = _METRICS
    minimum, resource, cache = 2, 'low', 'per-origin arrays'
    if family == 'base' and len(parts) == 3:
        minimum = {'Last month': 2, 'Mean 2': 2, 'Mean 3': 3,
                   'Mean 6': 6, 'Seasonal naive': 6,
                   'Damped trend': 6}.get(parts[1], 2)
        if parts[2].isdigit():
            minimum = max(minimum, int(parts[2]))
        label = f'آماری: {parts[1]} ({parts[2]})'
    elif family == 'base' and len(parts) == 2:  # valid synthetic test identifiers
        label = f'آماری: {parts[1]}'
    elif family == 'light' and len(parts) == 2:
        minimum = 12
        label = f'بنچمارک سبک: {parts[1]}'
    elif family == 'ml' and len(parts) == 3:
        minimum, resource, cache = 18, 'high: Global CatBoost', 'model+prediction fingerprint'
        label = f'CatBoost: {parts[1]} ({parts[2]})'
    elif family == 'price-trend' and len(parts) == 4:
        minimum, metrics = 6, ('value_1000_toman',)
        label = f'روند قیمت: {parts[3]} ({parts[1]}/{parts[2]})'
    elif family == 'price-volume' and len(parts) == 3:
        minimum, metrics = 18, ('value_1000_toman',)
        resource, cache = 'high: shared volume CatBoost', 'model+prediction fingerprint'
        label = f'حجم × قیمت: {parts[1]} ({parts[2]})'
    elif family == 'blend' and len(parts) >= 2:
        minimum = 18
        label = f'ترکیب وزنی {parts[1]}'
    else:
        raise ValueError(f'Unknown forecasting candidate identifier: {name}')
    return CandidateInfo(name, family, label, metrics, _FREQUENCIES,
                         minimum, resource, cache)


def candidate_registry(names):
    """Unique, serializable catalog of the candidates actually generated."""
    unique = sorted(set(names))
    return [candidate_info(name).record() for name in unique]



def eligible_candidates(by_origin, origins, metric, periodicity, row_count):
    """Validate the executable forecast grid against the candidate registry.

    The same candidates must be available at every selection origin, with
    identical (series, 3 horizons) arrays. An invalid or incompatible method
    cannot silently enter a model competition; rejected IDs are auditable.
    """
    if not origins or len(origins) != len(set(origins)):
        raise ValueError('Selection origins must be unique and nonempty.')
    if any(origin not in by_origin for origin in origins):
        raise ValueError('Missing candidate forecasts for a selection origin.')
    if metric not in _METRICS or periodicity not in _FREQUENCIES:
        raise ValueError('Unknown forecast metric or periodicity.')
    if row_count <= 0:
        raise ValueError('No series to validate.')
    shared = set.intersection(*(set(by_origin[origin]) for origin in origins))
    rejected = {}
    for name in sorted(set.union(*(set(by_origin[origin]) for origin in origins)) - shared):
        rejected[name] = 'not available at every validation origin'
    accepted = set()
    for name in sorted(shared):
        try:
            spec = candidate_info(name)
        except ValueError:
            rejected[name] = 'unregistered candidate identifier'
            continue
        if metric not in spec.metrics or periodicity not in spec.periodicities:
            rejected[name] = 'incompatible metric or periodicity'
            continue
        if any(origin + 1 < spec.min_history for origin in origins):
            rejected[name] = 'insufficient history for candidate'
            continue
        if any(np.asarray(by_origin[origin][name]).shape != (row_count, 3) or
               not np.isfinite(by_origin[origin][name]).all() or
               (by_origin[origin][name] < 0).any() for origin in origins):
            rejected[name] = 'invalid shape, negative value, NaN, or infinity'
            continue
        accepted.add(name)
    return accepted, rejected

def rolling_plan(cutoff, strategy='hierarchical', config=BacktestConfig(),
                 categories_available=True):
    """Strictly ordered origins with a wholly untouched later holdout.

    Short histories retain the established two-origin Global policy and have
    NO independently measured holdout. A new model is never validated on a
    target beyond cutoff. Origins are observation indexes, not month offsets.
    """
    if strategy not in ('hierarchical', 'global'):
        raise ValueError('Invalid forecast strategy.')
    if (config.selection_stride < config.max_horizon or config.max_selection_origins < 2
            or config.min_selection_origin < config.max_horizon):
        raise ValueError('Invalid chronological validation configuration.')
    if (strategy == 'hierarchical' and categories_available
            and cutoff >= config.min_independent_cutoff):
        holdout = cutoff - config.max_horizon
        selection = [holdout - config.selection_stride * i
                     for i in range(1, config.max_selection_origins + 1)]
        selection = sorted(o for o in selection if o >= config.min_selection_origin)
        if len(selection) >= 2:
            assert selection[-1] + config.max_horizon < holdout + 1
            assert holdout + config.max_horizon <= cutoff
            return {'selection': selection, 'holdout': holdout,
                    'independent': True, 'reason': 'rolling selection plus untouched holdout',
                    'configuration': asdict(config)}
    selection = [o for o in (cutoff-6, cutoff-3) if o >= 11]
    return {'selection': selection, 'holdout': None,
            'independent': False, 'reason': 'legacy Global fallback; independent holdout unavailable',
            'configuration': asdict(config)}


def choose_champion(losses, reference, guard=ChampionGuard()):
    """Choose a challenger only if its pooled AND per-origin gains are stable.

    losses: {candidate_id: abs-error vector over identical historical origins}.
    Includes the pre-existing baseline (best non-ML statistical candidate).
    If evidence is weak the reference remains the selected Recipe.
    """
    if reference not in losses:
        raise ValueError('Reference candidate missing from fair candidate comparison.')
    arrays = {name: np.asarray(value, dtype=float) for name, value in losses.items()}
    if not arrays:
        raise ValueError('Empty candidate comparison.')
    n = len(arrays[reference])
    if n == 0 or any(array.shape != (n,) or not np.isfinite(array).all()
                     or (array < 0).any() for array in arrays.values()):
        raise ValueError('Invalid or non-comparable per-origin candidate errors.')
    totals = {name: float(array.sum()) for name, array in arrays.items()}
    legacy = min(totals, key=lambda name: (totals[name], name))
    anchor = arrays[reference]
    baseline_loss = totals[reference]
    eligible = []
    required_wins = max(guard.min_selection_origins,
                        ceil(guard.min_winning_origin_fraction * n))
    if n >= guard.min_selection_origins and baseline_loss > 0:
        for name, values in arrays.items():
            if name == reference:
                continue
            gain = (baseline_loss - totals[name]) / baseline_loss
            win_count = int(np.count_nonzero(values < anchor - 1e-9))
            relative_worsening = np.divide(values - anchor, np.maximum(anchor, 1.0))
            if (gain >= guard.min_relative_improvement
                    and win_count >= required_wins
                    and float(relative_worsening.max()) <= guard.max_relative_origin_worsening):
                eligible.append(name)
    winner = min(eligible, key=lambda name: (totals[name], name)) if eligible else reference
    return winner, {
        'reference': reference, 'legacy_best': legacy,
        'selected': winner, 'candidate_count': len(arrays),
        'origins': n, 'required_origin_wins': required_wins,
        'selection_rule': 'pooled error reduction + origin stability + conservative reference fallback',
        'reason': 'stable improvement' if winner != reference else 'no challenger passed stability guard',
        'guard': asdict(guard),
        'reference_absolute_error': baseline_loss,
        'selected_absolute_error': totals[winner],
        'legacy_absolute_error': totals[legacy],
    }
