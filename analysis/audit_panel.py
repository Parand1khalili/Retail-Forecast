from pathlib import Path
from collections import defaultdict
import numpy as np,json
P=Path(__file__).parent

def load(name):
    z=np.load(P/(name+'.npz'))
    return z['meta'],z['values'],z['months'],z['source_rows']

m,a,months,src=load('aug_sales')
sel=m[:,13]=='Volume Sales(Kg)';m=m[sel];a=a[sel];src=src[sel]
r={'data_cutoff':str(months[-1]),'months':len(months),'dimensions':{},'checks':{},'benchmarks':{}}
for label,mask in [('cities',m[:,0]=='City'),('aggregate_24cities',m[:,1]=='24 Cities'),('outlet_types',m[:,0]=='OutletType'),('regions',m[:,0]=='Region')]:
    ar=a[mask];mr=m[mask]
    pos=(ar>0).sum(1);ever=pos>0
    r['dimensions'][label]={'rows':len(ar),'locations':len(set(mr[:,1])),'sku_descriptions':len(set(mr[:,3])),'allzero':int((~ever).sum()),'ever_positive':int(ever.sum()),'active_latest':int((ar[:,-1]>0).sum()),'zero_cells_pct':float((ar==0).mean()*100),'positive_months_among_ever_quantiles':np.quantile(pos[ever],[0,.25,.5,.75,1]).tolist(),'positive_last3':int((ar[:,-3:]>0).any(1).sum()),'everpositive_but_last6_zero':int((ever&~(ar[:,-6:]>0).any(1)).sum())}

total=a[m[:,1]=='24 Cities'].sum(0)
for label,mask in [('citysum',m[:,0]=='City'),('channelsum',m[:,0]=='OutletType'),('regionsum',m[:,0]=='Region')]:
    sm=a[mask].sum(0)
    r['checks'][label]={'max_abs_relative_diff':float(np.max(np.abs(sm-total)/total)),'latest':float(sm[-1]),'total24_latest':float(total[-1])}

def aggregate(mask,cols):
    d=defaultdict(lambda:np.zeros(len(months)))
    for mi,ai in zip(m[mask],a[mask]): d[tuple(mi[c] for c in cols)]+=ai
    return np.array(list(d.values())),list(d)

brand_city,bc_keys=aggregate(m[:,0]=='City',[1,10])
sku_city,sc_keys=aggregate(m[:,0]=='City',[1,3,10])
r['dimensions']['brand_city']={'series':len(brand_city),'ever_positive':int((brand_city>0).any(1).sum()),'active_latest':int((brand_city[:,-1]>0).sum())}

# Six rolling origins. Each origin forecasts BOTH months without using either future actual.
origins=list(range(len(months)-8,len(months)-2))
for label,ar in [('brand_city',brand_city),('sku_city',sku_city),('total24',total[None,:])]:
    res={}
    for method in ['last_month','mean_last3','same_month_last_year']:
        all_y=[];all_p=[];by_h={1:[[],[]],2:[[],[]]}
        for t in origins:
            for h in [1,2]:
                y=ar[:,t+h]
                if method=='last_month': pred=ar[:,t]
                elif method=='mean_last3': pred=ar[:,t-2:t+1].mean(1)
                else: pred=ar[:,t+h-12]
                all_y.extend(y);all_p.extend(pred)
                by_h[h][0].extend(y);by_h[h][1].extend(pred)
        y=np.array(all_y);p=np.array(all_p)
        res[method]={'wape_pct':float(np.abs(p-y).sum()/y.sum()*100),'bias_pct':float((p-y).sum()/y.sum()*100),'by_h':{str(h):float(np.abs(np.array(v[1])-np.array(v[0])).sum()/np.sum(v[0])*100) for h,v in by_h.items()}}
    r['benchmarks'][label]=res
r['benchmark_origins']=[str(months[t]) for t in origins]
r['benchmark_target_span']=[str(months[origins[0]+1]),str(months[origins[-1]+2])]
r['total24_history']={str(t):float(v) for t,v in zip(months,total)}
r['august_yoy']={}
fm,fa,ft,fs=load('aug_sales')
for metric in ['Volume Sales(Kg)','Value Sales (1,000 Toomans)']:
    vals=fa[(fm[:,1]=='24 Cities')&(fm[:,13]==metric)].sum(0)
    v25=vals[list(ft).index("Aug' 2025")];v26=vals[-1]
    r['august_yoy'][metric]={'aug25':float(v25),'aug26':float(v26),'change_pct':float((v26/v25-1)*100)}

if (P/'jun_sales.npz').exists():
    jm,ja,jmonths,jsrc=load('jun_sales');msk=jm[:,13]=='Volume Sales(Kg)';jm=jm[msk];ja=ja[msk]
    common=sorted(set(months)&set(jmonths),key=lambda x:list(months).index(x))
    def keyed(meta,values):
        d=defaultdict(lambda:np.zeros(values.shape[1]))
        for mi,ai in zip(meta,values): d[tuple(mi[:13])]+=np.nan_to_num(ai)
        return d
    ad=keyed(m,a);jd=keyed(jm,ja);keys=set(ad)&set(jd)
    ix=[list(months).index(t) for t in common];jx=[list(jmonths).index(t) for t in common]
    av=np.array([ad[k][ix] for k in sorted(keys)]);jv=np.array([jd[k][jx] for k in sorted(keys)])
    delta=np.abs(av-jv);tol=np.maximum(1e-6,np.abs(av)*1e-8)
    r['old_vs_new']={'jun_months':len(jmonths),'jun_period':[str(jmonths[0]),str(jmonths[-1])],'common_months':len(common),'common_keys':len(keys),'aug_only_keys':len(set(ad)-set(jd)),'jun_only_keys':len(set(jd)-set(ad)),'changed_cells':int((delta>tol).sum()),'compared_cells':int(delta.size),'relative_absolute_revision_pct':float(delta.sum()/np.abs(av).sum()*100)}
    jj=ja[jm[:,1]=='24 Cities'];jtot=np.nansum(jj,axis=0)
    r['old_vs_new']['total24_common_revision_pct']=float(np.abs(total[ix]-jtot[jx]).sum()/total[ix].sum()*100)
    r['old_vs_new']['jun_zero_fraction_first_month']=float((ja[:,0]==0).mean())
    r['old_vs_new']['jun_missing_fraction_first_month']=float(np.isnan(ja[:,0]).mean())

if (P/'pivot_sales.npz').exists():
    pm,pa,pt,psrc=load('pivot_sales')
    r['pivot_comparison']={'shape':[len(pm),pa.shape[1]],'same_months':bool(np.array_equal(pt,months))}
    # Compare keyed values because row order and stale used range may differ.
    r['pivot_comparison'].update(same_row_metadata=bool(np.array_equal(pm,fm)),same_sales_values=bool(np.allclose(pa,fa,atol=1e-6,rtol=1e-8,equal_nan=True)))

(P/'audit_summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2))
