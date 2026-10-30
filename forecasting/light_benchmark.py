"""Small, category-independent monthly benchmark with nested rolling selection.

All scores are retrospective: these source snapshots were already inspected.
No CatBoost dependency; no source edits; output is separate from issued forecasts.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from app_core import Dataset, ROOT

VERSION = 'light-monthly-v1'


def candidates(volume, value, metric, horizon):
    y = volume if metric == 'volume_kg' else value
    last = y[:, -1]
    out = {'last': last, 'mean2': y[:, -2:].mean(1),
           'mean3': y[:, -3:].mean(1), 'seasonal12': y[:, -12+horizon-1]}
    for alpha in [.2, .5, .8]:
        level = y[:, 0].copy()
        for j in range(1, y.shape[1]):
            level += alpha * (y[:, j] - level)
        out[f'ses{alpha}'] = level
    # Croston SBA: smooth positive sizes and intervals separately; zeros retained.
    size = np.zeros(len(y)); interval = np.ones(len(y)); gap = np.zeros(len(y))
    seen = np.zeros(len(y), dtype=bool)
    for j in range(y.shape[1]):
        gap += 1
        positive = y[:, j] > 0
        first = positive & ~seen
        size[first] = y[first, j]; interval[first] = gap[first]
        update = positive & seen
        size[update] += .1 * (y[update, j] - size[update])
        interval[update] += .1 * (gap[update] - interval[update])
        seen |= positive
        gap[positive] = 0
    out['croston_sba'] = .95 * size / interval
    out['equal_blend'] = (out['last'] + out['mean3'] + out['ses0.5']) / 3
    if metric == 'value_1000_toman':
        prices = np.divide(value, volume, out=np.full_like(value, np.nan), where=volume > 0)
        prices = pd.DataFrame(prices).ffill(axis=1).fillna(0).to_numpy()
        # Fixed robust local price trend; no category-specific shock threshold.
        recent = prices[:, -7:]
        change = np.divide(recent[:, 1:], recent[:, :-1],
                           out=np.ones_like(recent[:, 1:]), where=recent[:, :-1] > 0)
        growth = np.clip(np.median(change, axis=1), .9, 1.15)
        for name, qty in [('last', volume[:, -1]), ('mean2', volume[:, -2:].mean(1))]:
            out[f'price_volume_{name}'] = qty * prices[:, -1] * growth ** horizon
    return {name: np.maximum(pred, 0) for name, pred in out.items()}


def score(actual, prediction):
    den = float(actual.sum())
    return {'wape': float(abs(prediction-actual).sum()/den) if den else None,
            'bias': float((prediction-actual).sum()/den) if den else None}


def evaluate(d, output):
    if len(d.months) < 30:
        raise ValueError('At least 30 complete monthly observations required for this benchmark.')
    begun = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    rows = []; selections = []; summaries = []; future = {}
    # All horizons share origins and available three-month targets.
    origins = list(range(max(23, len(d.months)-15), len(d.months)-3))
    for metric in ['volume_kg', 'value_1000_toman']:
        y = d.metric(metric)
        collected = {}
        future[metric] = []
        for h in [1, 2, 3]:
            cache = {}
            def predict(o):
                if o not in cache:
                    cache[o] = candidates(d.volume[:, :o+1], d.value[:, :o+1], metric, h)
                return cache[o]
            for o in origins + [len(d.months)-1]:
                # Six earlier origins whose targets were known at this origin.
                inner = list(range(o-h-5, o-h+1))
                errors = {name: 0. for name in predict(o)}
                for v in inner:
                    for name, p in predict(v).items():
                        errors[name] += float(abs(p-y[:, v+h]).sum())
                winner = min(errors, key=errors.get)
                selections.append({'metric': metric, 'horizon': h, 'origin': d.months[o],
                                   'method': winner, 'selection_origins': [d.months[v] for v in inner],
                                   'latest_selection_target': d.months[inner[-1]+h]})
                if o == len(d.months)-1:
                    future[metric].append(predict(o)[winner]); continue
                for name, p in {**predict(o), 'rolling_selected': predict(o)[winner]}.items():
                    actual = y[:, o+h]
                    rows.append({'metric': metric, 'horizon': h, 'origin': d.months[o],
                                 'target': d.months[o+h], 'method': name, **score(actual, p)})
                    collected.setdefault((h, name), []).append((actual, p))
        for (h, name), values in collected.items():
            actual = np.stack([v[0] for v in values], axis=1)
            pred = np.stack([v[1] for v in values], axis=1)
            summaries.append({'metric': metric, 'horizon': h, 'method': name,
                              'origins': len(values), **score(actual, pred)})
    pd.DataFrame(rows).to_csv(output/'rolling.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False, encoding='utf-8-sig')
    (output/'selection.json').write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output/'future.npz', meta=d.meta,
                        months=np.array([str(pd.Period(d.months[-1])+h) for h in [1, 2, 3]]),
                        **{m: np.stack(v, axis=1) for m, v in future.items()})
    audit = {'version': VERSION, 'source': d.source, 'months': len(d.months), 'series': len(d.meta),
             'source_end': d.months[-1], 'volume_unit': d.volume_unit, 'value_unit': d.value_unit,
             'evaluation_origins': [d.months[o] for o in origins],
             'seconds': time.perf_counter()-begun,
             'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'data_sha256': hashlib.sha256(d.meta.tobytes()+d.volume.tobytes()+d.value.tobytes()).hexdigest(),
             'limitations': 'Retrospective current-taxonomy data; overlapping RA windows; no untouched test. '
                            'Selection is global per metric/horizon, not per individual product. '
                            'Future outputs are experimental, not a replacement for issued forecasts.'}
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=True))
    print(pd.DataFrame(summaries).query("method in ['last','rolling_selected']").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help='Monthly RA xlsx; omitted = yoghurt cache')
    parser.add_argument('--mapping', type=Path, help='Explicit schema/measure/unit mapping JSON')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/light_benchmark')
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding='utf-8')) if args.mapping else None
    evaluate(Dataset.file(args.input, mapping) if args.input else Dataset.builtin(), args.output)
