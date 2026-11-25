"""Retrospective outer rolling-origin audit of the actual production policies.

The source workbook is not modified. Current-snapshot metadata is NOT a
historical data vintage. Only series with positive Volume or Value by each
outer origin enter that origin's panel; later/unobserved series score at zero.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from app_core import Dataset
from auto_engine import automatic_forecast, VERSION, POLICY_VERSION
from benchmark_levels import error_metrics

METRICS=('volume_kg','value_1000_toman')
STRATEGIES=('global','hierarchical')
SCOPE='current_snapshot_known_series_proxy_live_policy'


def outer_origins(periods, count=3):
    """Recent, non-overlapping three-period test blocks; at least 18 inputs."""
    if not 2<=count<=6:
        raise ValueError('Request 2 to 6 outer origins.')
    origins=list(range(periods-4,16,-3))[:count]
    if len(origins)<2:
        raise ValueError('At least 24 observations are required for two complete outer folds.')
    return sorted(origins)


def prefix_panel(dataset, origin):
    """Do not give the policy future dates, sales or identities first seen later."""
    q=dataset.volume[:,:origin+1]
    v=dataset.value[:,:origin+1]
    known=(q>0).any(axis=1)|(v>0).any(axis=1)
    if not known.any():
        raise ValueError(f'No observed positive sales at outer origin {dataset.months[origin]}.')
    return replace(dataset,meta=dataset.meta[known],volume=q[known],value=v[known],
                   months=list(dataset.months[:origin+1])),known


def run_walk_forward(dataset, metric, count=3, *, include_adaptive=False):
    """Re-run selection and inference at every outer origin; never tune on tests."""
    if metric not in METRICS:
        raise ValueError('Unknown sales metric.')
    dims=dataset.all_dimensions()
    if 'Market' in dims and len(set(dataset.meta[:,dims.index('Market')]))!=1:
        raise ValueError('Select one Market before running the audit.')
    if len(dataset.report_grain_levels())>1:
        raise ValueError('Select one report grain before running the audit.')
    truth=dataset.metric(metric)
    rows=[]
    for origin in outer_origins(len(dataset.months),count):
        prefix,known=prefix_panel(dataset,origin)
        for h in (1,2,3):
            forecasts=[]
            for strategy in (*STRATEGIES,'adaptive') if include_adaptive else STRATEGIES:
                # This is the live entry point, not a locked Recipe or a
                # reimplementation of the Global/Hierarchical algorithms.
                prediction,recipe=automatic_forecast(prefix,metric,h,strategy=strategy)
                prediction=np.asarray(prediction,dtype=float)
                if (prediction.shape!=(int(known.sum()),) or
                    not np.isfinite(prediction).all() or (prediction<0).any()):
                    raise ValueError('Invalid production prediction at outer origin.')
                full=np.zeros(len(known),dtype=float)
                full[known]=prediction
                category=recipe.get('category_selection') or {}
                forecasts.append((strategy,full,recipe['name'],
                    bool((recipe.get('price_guard') or {}).get('active')),
                    category.get('selection_origins',recipe.get('validation_origins',[])),
                    category.get('usage',{})))
            forecasts.append(('last_observation',truth[:,origin],'Last observation',False,[],{}))
            # Only access outer targets after every policy has forecast this horizon.
            actual=truth[:,origin+h]
            for strategy,pred,name,guard,selection,usage in forecasts:
                stats=error_metrics(actual,pred)
                market_error=abs(stats['predicted']-stats['actual'])
                rows.append({'scope':SCOPE,'engine_version':VERSION,'policy_version':POLICY_VERSION,
                    'metric':metric,'periodicity':dataset.periodicity,
                    'strategy':strategy,'origin':dataset.months[origin],
                    'target':dataset.months[origin+h],'horizon':h,'recipe':name,
                    'price_guard_active':guard,'inner_selection_origins':json.dumps(selection),
                    'recipe_usage':json.dumps(usage,ensure_ascii=False),
                    'known_series':int(known.sum()),'unobserved_series':int((~known).sum()),
                    'unobserved_actual':float(actual[~known].sum()),
                    **stats,'market_absolute_error':market_error,
                    'market_ape':market_error/stats['actual'] if stats['actual']>0 else None})
    return rows


def summarize(rows):
    """Pool numerators and denominators, not averages of percentage errors."""
    result=[]
    for strategy in (*STRATEGIES,'adaptive','last_observation'):
        for h in (1,2,3):
            group=[r for r in rows if r['strategy']==strategy and r['horizon']==h]
            if not group:
                continue
            actual=sum(r['actual'] for r in group)
            predicted=sum(r['predicted'] for r in group)
            absolute=sum(r['absolute_error'] for r in group)
            market_absolute=sum(r['market_absolute_error'] for r in group)
            cells=sum(r['cells'] for r in group)
            result.append({'scope':SCOPE,'engine_version':VERSION,'policy_version':POLICY_VERSION,
                'metric':group[0]['metric'],'periodicity':group[0]['periodicity'],
                'strategy':strategy,'horizon':h,'origins':len(group),'cells':cells,
                'actual':actual,'predicted':predicted,'absolute_error':absolute,
                'wape':absolute/actual if actual>0 else None,'mae':absolute/cells if cells else None,
                'bias':(predicted-actual)/actual if actual>0 else None,
                'market_absolute_error':market_absolute,
                'market_wape':market_absolute/actual if actual>0 else None,
                'unobserved_actual':sum(r['unobserved_actual'] for r in group),
                'price_guard_origins':sum(r['price_guard_active'] for r in group)})
    return result


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook',type=Path)
    parser.add_argument('--periodicity',choices=('monthly','bimonthly'),default='monthly')
    parser.add_argument('--market',help='One geographic reporting level, e.g. City')
    parser.add_argument('--grain',help='One Element Caption reporting grain when required')
    parser.add_argument('--metric',choices=METRICS,default='volume_kg')
    parser.add_argument('--origins',type=int,choices=range(2,7),default=3)
    parser.add_argument('--adaptive',action='store_true',help='Compare experimental brand/city and MAE policy against stable policies.')
    parser.add_argument('--output',type=Path,default=Path('outputs/live_policy_walk_forward.csv'))
    args=parser.parse_args()
    dataset=Dataset.file(args.workbook,{'periodicity':args.periodicity})
    dataset=dataset.for_market(args.market).for_grain(args.grain)
    rows=run_walk_forward(dataset,args.metric,args.origins,include_adaptive=args.adaptive)
    summary_path=args.output.with_name(args.output.stem+'_summary.csv')
    write_csv(args.output,rows)
    write_csv(summary_path,summarize(rows))
    print(f'Outer folds: {args.output}; pooled results: {summary_path}')
    print('Current snapshot / known-series proxy, not vintage-as-of data; no tuning.')


if __name__=='__main__':
    main()
