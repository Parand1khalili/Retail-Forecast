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
from auto_engine import dataset_policy, candidates
from category_selection import forecast_with_recipes
from benchmark_levels import compare_levels


def report(dataset, metric, output, detailed_output=None):
    policy=dataset_policy(dataset,metric,len(dataset.months)-1,'hierarchical')
    audit=policy.get('_holdout') or {}
    origin=audit.get('origin')
    dims=dataset.all_dimensions()
    market=(str(dataset.meta[0,dims.index('Market')]) if 'Market' in dims else 'N/A')
    records=[]
    detailed=[]
    held=None
    holdout_index=dataset.months.index(origin) if origin is not None else None
    if holdout_index is not None:
        # This uses the SAME historical holdout but never reselects/tunes recipes.
        held=candidates(dataset,metric,holdout_index)
    for horizon in ('1','2','3'):
        stats=audit.get('horizons',{}).get(horizon)
        if stats:
            row={'metric':metric,'market':market,
                 'periodicity':dataset.periodicity,'horizon':horizon,
                 'audit_type':'untouched_holdout','origin':origin,
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
                    detailed.append({'metric':metric,'market':market,
                                     'horizon':horizon,'origin':origin,
                                     'target':dataset.months[holdout_index+int(horizon)],
                                     **item})
        else:
            row={'metric':metric,'market':market,'periodicity':dataset.periodicity,
                 'horizon':horizon,'audit_type':'no_independent_holdout',
                 'origin':'','selection_origins':','.join(policy['_selection_origins']),
                 'global_recipe':policy[horizon]['name'],'selection_only':True}
        records.append(row)
    fields=['metric','market','periodicity','horizon','audit_type','origin',
            'selection_origins','global_recipe','global_wape','hierarchical_wape',
            'legacy_global_recipe','legacy_global_wape',
            'legacy_global_mae','legacy_global_bias',
            'global_mae','hierarchical_mae','global_bias','hierarchical_bias',
            'relative_absolute_error_reduction','relative_reduction_vs_legacy',
            'groups_improved','groups_deteriorated',
            'groups_unchanged','series_using_override','selection_only']
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
