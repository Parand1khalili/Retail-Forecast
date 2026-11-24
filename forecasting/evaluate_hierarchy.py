"""Reproducible selection-vs-independent-holdout audit for a private RA master.

Example:
  python forecasting/evaluate_hierarchy.py data.xlsx --periodicity monthly --market All \
      --metric volume_kg --output outputs/hierarchy_audit.csv

Selection results and the independent holdout are labeled separately. This
tool never publishes or modifies confidential source Excel records.
"""
import argparse
import csv
from pathlib import Path
from app_core import Dataset
from auto_engine import dataset_policy, candidates, guarded_money, automatic_forecast
from category_selection import forecast_with_recipes
from benchmark_levels import compare_levels, error_metrics


def report(dataset, metric, output, detailed_output=None):
    policy=dataset_policy(dataset,metric,len(dataset.months)-1,'hierarchical')
    audit=policy.get('_holdout') or {}
    origin=audit.get('origin')
    dims=dataset.all_dimensions()
    market=(str(dataset.meta[0,dims.index('Market')]) if 'Market' in dims else 'N/A')
    records=[]
    detailed=[]
    held=None
    guard_prediction=None
    guard_info=None
    holdout_index=dataset.months.index(origin) if origin is not None else None
    if holdout_index is not None:
        # This uses the SAME historical holdout but never reselects/tunes recipes.
        held=candidates(dataset,metric,holdout_index)
        if metric=='value_1000_toman':
            # Reconstruct the OLD forced override at the actual holdout origin.
            # The new live policy is independently replayed below.
            guard_prediction,guard_info=guarded_money(dataset,holdout_index)
    for horizon in ('1','2','3'):
        stats=audit.get('horizons',{}).get(horizon)
        if stats:
            row={'metric':metric,'market':market,
                 'periodicity':dataset.periodicity,'horizon':horizon,
                 'audit_type':'untouched_holdout','origin':origin,
                 'audit_scope':'locked_recipes_common_selection_origins',
                 'price_guard_active_at_holdout':(
                     bool(guard_info['active']) if guard_info is not None else None),
                 'selection_origins':','.join(policy['_selection_origins']),
                 'global_recipe':policy[horizon]['name'],
                 'global_wape':stats['global']['wape'],
                 'legacy_global_wape':stats['legacy_global']['wape'],
                 'legacy_global_mae':stats['legacy_global']['mae'],
                 'legacy_global_bias':stats['legacy_global']['bias'],
                 'legacy_global_recipe':stats['legacy_recipe'],
                 'hierarchical_wape':stats['hierarchical']['wape'],
                 'global_mae':stats['global']['mae'],
                 'hierarchical_mae':stats['hierarchical']['mae'],
                 'global_bias':stats['global']['bias'],
                 'hierarchical_bias':stats['hierarchical']['bias'],
                 'relative_absolute_error_reduction':stats['relative_error_reduction'],
                 'relative_reduction_vs_legacy':stats['relative_error_reduction_vs_legacy'],
                 'groups_improved':stats['groups_improved'],
                 'groups_deteriorated':stats['groups_deteriorated'],
                 'groups_unchanged':stats['groups_unchanged'],
                 'series_using_override':stats['series_using_override'],
                 'selection_only':False}
            if guard_prediction is not None:
                actual=dataset.metric(metric)[:,holdout_index+int(horizon)]
                guard_stats=error_metrics(actual,guard_prediction[:,int(horizon)-1])
                row.update(price_guard_wape=guard_stats['wape'],
                           price_guard_mae=guard_stats['mae'],
                           price_guard_bias=guard_stats['bias'],
                           price_guard_absolute_error=guard_stats['absolute_error'])
            live_details={}
            if metric=='value_1000_toman':
                h=int(horizon)
                actual=dataset.metric(metric)[:,holdout_index+h]
                # Replay the exact production inference at the historical
                # cutoff, NOT the recipe fitted using later observations.
                live, calibration=automatic_forecast(dataset,metric,h,cutoff=holdout_index)
                live_global,_=automatic_forecast(dataset,metric,h,cutoff=holdout_index,
                                                  strategy='global')
                # The previous version forced the specialist for ALL strategies
                # on an active shock; otherwise both versions are identical.
                previous=guard_prediction[:,h-1] if guard_prediction is not None else live
                old_stats=error_metrics(actual,previous)
                new_stats=error_metrics(actual,live)
                global_stats=error_metrics(actual,live_global)
                regime=calibration.get('price_regime') or {}
                row.update(live_audit_scope='cutoff_safe_policy_replay',
                           live_hierarchical_wape=new_stats['wape'],
                           live_global_wape=global_stats['wape'],
                           previous_live_wape=old_stats['wape'],
                           live_hierarchical_mae=new_stats['mae'],
                           previous_live_mae=old_stats['mae'],
                           live_hierarchical_bias=new_stats['bias'],
                           previous_live_bias=old_stats['bias'],
                           live_relative_reduction_vs_previous=(
                               1-new_stats['absolute_error']/old_stats['absolute_error']
                               if old_stats['absolute_error']>0 else None),
                           live_recipe=calibration['name'],
                           live_regime_reason=regime.get('reason','normal'),
                           live_shock_origins=','.join(regime.get('origins',[])),
                           live_shock_events=len(regime.get('origins',[])),
                           live_specialist_series=regime.get('specialist_series',0))
                live_details={(item['level'],item['group']):item for item in
                              compare_levels(dataset,actual,live_global,live,holdout_index,
                                             legacy=previous)}
            entry=policy[horizon]
            legacy=policy['_legacy_global'][horizon]
            needed={name for recipe in (entry,legacy)
                    for name,_ in recipe['components']}
            needed.update(name for node in policy['_hierarchy']['overrides'].get(horizon,[])
                          for name,_ in node['components'])
            if needed.issubset(held):
                def weighted(recipe):
                    return sum(weight*held[name][:,int(horizon)-1]
                               for name,weight in recipe['components'])
                global_prediction=weighted(entry)
                legacy_prediction=weighted(legacy)
                hierarchical_prediction,_,_=forecast_with_recipes(
                    dataset,int(horizon),held,entry,policy['_hierarchy'])
                target_actual=dataset.metric(metric)[:,holdout_index+int(horizon)]
                for item in compare_levels(dataset,target_actual,global_prediction,
                                           hierarchical_prediction,holdout_index,
                                           legacy=legacy_prediction):
                    live_item=live_details.get((item['level'],item['group']))
                    extra=({'live_'+k:v for k,v in live_item.items()
                            if k not in ('level','group','series','actual')}
                           if live_item is not None else {})
                    detailed.append({'metric':metric,'market':market,
                                     'horizon':horizon,'origin':origin,
                                     'target':dataset.months[holdout_index+int(horizon)],
                                     **item,**extra})
        else:
            row={'metric':metric,'market':market,'periodicity':dataset.periodicity,
                 'horizon':horizon,'audit_type':'no_independent_holdout',
                 'origin':'','audit_scope':'selection_only',
                 'selection_origins':','.join(policy['_selection_origins']),
                 'global_recipe':policy[horizon]['name'],'selection_only':True}
        records.append(row)
    fields=['metric','market','periodicity','horizon','audit_type','origin',
            'selection_origins','global_recipe','global_wape','hierarchical_wape',
            'legacy_global_recipe','legacy_global_wape',
            'legacy_global_mae','legacy_global_bias',
            'global_mae','hierarchical_mae','global_bias','hierarchical_bias',
            'relative_absolute_error_reduction','relative_reduction_vs_legacy',
            'groups_improved','groups_deteriorated',
            'groups_unchanged','series_using_override','selection_only',
            'audit_scope','price_guard_active_at_holdout','price_guard_wape',
            'price_guard_mae','price_guard_bias','price_guard_absolute_error',
            'live_audit_scope','live_hierarchical_wape','live_global_wape','previous_live_wape',
            'live_hierarchical_mae','previous_live_mae','live_hierarchical_bias','previous_live_bias',
            'live_relative_reduction_vs_previous','live_recipe','live_regime_reason',
            'live_shock_origins','live_shock_events','live_specialist_series']
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if detailed:
        details=Path(detailed_output) if detailed_output is not None else (
            output.with_name(output.stem+'_by_group.csv'))
        details.parent.mkdir(parents=True,exist_ok=True)
        columns=list(detailed[0])
        with details.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows(detailed)
    return records


def main():
    parser=argparse.ArgumentParser(description='Private hierarchical forecast holdout audit')
    parser.add_argument('workbook',type=Path)
    parser.add_argument('--periodicity',choices=('monthly','bimonthly'),default='monthly')
    parser.add_argument('--market',help='Select one disjoint Market level, e.g. All or City')
    parser.add_argument('--grain',help='Element Caption when source mixes SKU and Product')
    parser.add_argument('--metric',choices=('volume_kg','value_1000_toman'),
                        default='volume_kg')
    parser.add_argument('--output',type=Path,default=Path('outputs/hierarchy_audit.csv'))
    parser.add_argument('--detailed-output',type=Path,default=None,
                        help='Optional per-category, Brand, series, sparse-demand audit CSV')
    args=parser.parse_args()
    dataset=Dataset.file(args.workbook,{'periodicity':args.periodicity})
    dataset=dataset.for_market(args.market).for_grain(args.grain)
    rows=report(dataset,args.metric,args.output,args.detailed_output)
    for row in rows:
        print('H'+row['horizon'],row['audit_type'],
              'legacy=',row.get('legacy_global_wape'),
              'guarded_global=',row.get('global_wape'),
              'hierarchical=',row.get('hierarchical_wape'),
              'origin=',row['origin'])
    print('Report:',args.output.resolve())


if __name__=='__main__':
    main()
