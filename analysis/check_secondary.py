from pathlib import Path
import numpy as np,json
from collections import defaultdict
P=Path(__file__).parent

def panel(name):
    z=np.load(P/(name+'.npz'));m=z['meta'];a=z['values'];sel=m[:,13]=='Volume Sales(Kg)'
    return m[sel],a[sel],z['months']

def keyed(m,a,cols):
    d=defaultdict(lambda:np.zeros(a.shape[1]))
    for mi,ai in zip(m,a):d[tuple(mi[c] for c in cols)]+=ai
    return d

m,a,t=panel('aug_sales');pm,pa,pt=panel('pivot_sales')
d=keyed(m,a,list(range(13)));pd=keyed(pm,pa,list(range(13)))
common=set(d)&set(pd);ix=sorted(common)
da=np.array([d[k] for k in ix]);db=np.array([pd[k] for k in ix])
r={'pivot_canonical_comparison':{'common_keys':len(common),'master_only':len(set(d)-set(pd)),'pivot_only':len(set(pd)-set(d)),'changed_cells':int((~np.isclose(da,db,rtol=1e-8,atol=1e-6)).sum()),'compared_cells':int(da.size)}}

result={}
for label,cols in [('brand_city',[1,10]),('sku_city',[1,3,10])]:
    v=keyed(m[m[:,0]=='City'],a[m[:,0]=='City'],cols);ar=np.array(list(v.values()))
    res={}
    for method in ['last_month','mean_last3','same_month_last_year']:
        err=actual=0.
        for origin in range(13,len(t)-2):
            for h in [1,2]:
                y=ar[:,origin+h]
                if method=='last_month':p=ar[:,origin]
                elif method=='mean_last3':p=ar[:,origin-2:origin+1].mean(1)
                else:p=ar[:,origin+h-12]
                err+=np.abs(p-y).sum();actual+=y.sum()
        res[method]=err/actual*100
    result[label]=res
r['extended_benchmark']={'origins':18,'origin_span':[str(t[13]),str(t[-3])],'target_span':[str(t[14]),str(t[-1])],'wape_pct':result}
(P/'secondary_checks.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2))
