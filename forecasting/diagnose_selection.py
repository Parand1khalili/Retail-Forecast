"""Inspect temporal stability using development months only, not Jun-Aug labels."""
import json,numpy as np,pandas as pd
from app_core import Dataset
from auto_engine import base_candidates,candidates,OUT
d=Dataset.builtin();dates=pd.PeriodIndex(d.months,freq='M')
rows=[]
for origin in range(dates.get_loc(pd.Period('2024-03')),dates.get_loc(pd.Period('2026-02'))+1):
    pp=base_candidates(d,'value_1000_toman',origin)
    for h in [1,2,3]:
        truth=d.value[:,origin+h]
        ref=abs(pp['base|Last month|2'][:,h-1]-truth).sum()/truth.sum()
        for name,p in pp.items():
            err=abs(p[:,h-1]-truth).sum()/truth.sum()
            rows.append({'origin':d.months[origin],'horizon':h,'name':name,'wape':err,'regret':err-ref})
df=pd.DataFrame(rows);df.to_csv(OUT/'baseline_stability.csv',index=False,encoding='utf-8-sig')
summary=df.groupby(['horizon','name']).agg(mean=('wape','mean'),std=('wape','std'),worst=('wape','max'),regret=('regret','mean'),worst_regret=('regret','max'),wins=('regret',lambda x:int((x<0).sum()))).reset_index()
summary['risk_score']=summary['mean']+.5*summary['std']
for h in [1,2,3]:
    print('HORIZON',h,flush=True)
    print(summary[summary.horizon==h].sort_values('risk_score').head(8).to_string(index=False),flush=True)
ml=[]
for label in ['2025-05','2025-09','2026-02']:
    o=dates.get_loc(pd.Period(label));pp=candidates(d,'value_1000_toman',o)
    for h in [1,2,3]:
        truth=d.value[:,o+h];ref=abs(pp['base|Last month|2'][:,h-1]-truth).sum()/truth.sum()
        for name,p in pp.items():
            err=abs(p[:,h-1]-truth).sum()/truth.sum()
            ml.append({'origin':label,'horizon':h,'name':name,'wape':err,'regret_vs_last':err-ref})
pd.DataFrame(ml).to_csv(OUT/'ml_stability.csv',index=False,encoding='utf-8-sig')
print('Development stability diagnostics saved.',flush=True)
