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
from auto_engine import dataset_policy


def report(dataset, metric, output):
    policy=dataset_policy(dataset,metric,len(dataset.months)-1,'hierarchical')
    audit=policy.get('_holdout') or {}
    origin=audit.get('origin')
    dims=dataset.all_dimensions()
    market=(str(dataset.meta[0,dims.index('Market')]) if 'Market' in dims else 'N/A')
    records=[]
    for horizon in ('1','2','3'):
        stats=audit.get('horizons',{}).get(horizon)
        if stats:
            row={'metric':metric,'market':market,
                 'periodicity':dataset.periodicity,'horizon':horizon,
                 'audit_type':'untouched_holdout','origin':origin,
                 'selection_origins':','.join(policy['_selection_origins']),
                 'global_recipe':policy[horizon]['name'],
                 'global_wape':stats['global']['wape'],
                 'hierarchical_wape':stats['hierarchical']['wape'],
                 'global_mae':stats['global']['mae'],
                 'hierarchical_mae':stats['hierarchical']['mae'],
                 'global_bias':stats['global']['bias'],
                 'hierarchical_bias':stats['hierarchical']['bias'],
                 'relative_absolute_error_reduction':stats['relative_error_reduction'],
                 'groups_improved':stats['groups_improved'],
                 'groups_deteriorated':stats['groups_deteriorated'],
                 'groups_unchanged':stats['groups_unchanged'],
                 'series_using_override':stats['series_using_override'],
                 'selection_only':False}
        else:
            row={'metric':metric,'market':market,'periodicity':dataset.periodicity,
                 'horizon':horizon,'audit_type':'no_independent_holdout',
                 'origin':'','selection_origins':','.join(policy['_selection_origins']),
                 'global_recipe':policy[horizon]['name'],'selection_only':True}
        records.append(row)
    fields=['metric','market','periodicity','horizon','audit_type','origin',
            'selection_origins','global_recipe','global_wape','hierarchical_wape',
            'global_mae','hierarchical_mae','global_bias','hierarchical_bias',
            'relative_absolute_error_reduction','groups_improved','groups_deteriorated',
            'groups_unchanged','series_using_override','selection_only']
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
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
    args=parser.parse_args()
    dataset=Dataset.file(args.workbook,{'periodicity':args.periodicity})
    dataset=dataset.for_market(args.market).for_grain(args.grain)
    rows=report(dataset,args.metric,args.output)
    for row in rows:
        print('H'+row['horizon'],row['audit_type'],
              'global=',row.get('global_wape'),
              'hierarchical=',row.get('hierarchical_wape'),
              'origin=',row['origin'])
    print('Report:',args.output.resolve())


if __name__=='__main__':
    main()
