"""Three-month fixed-origin retail-audit volume backtest. Source workbooks are read-only.

Validation origins and model presets are fixed before the Jun-Aug 2026 holdout.
Only lagged sales and unit prices, static taxonomy and known calendar features are used.
Source snapshots use the customer's current classification, not archived publication vintages.
"""
from pathlib import Path
import sys,json,hashlib,time,calendar
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/model_deps'))
from catboost import CatBoostRegressor

OUT=ROOT/'outputs/yoghurt_backtest'
OUT.mkdir(parents=True,exist_ok=True)
COLS=[1,11,10,12,4,5,6,7]
DIMS=['City','Company','Brand','SubBrand','Subcategory','Subcategory2','Subcategory3','Subcategory4']
BASELINES=['Last month','Mean 3 months','Seasonal naive','Seasonal adjusted','Exponential smoothing']
ML=['CatBoost residual','CatBoost scaled residual']
PRESETS={
    'iterations':200,'depth':5,'learning_rate':0.05,'l2_leaf_reg':8,
    'loss_function':'RMSE','random_seed':42,'thread_count':4,
    'verbose':False,'allow_writing_files':False,'has_time':True,'one_hot_max_size':32,
}

def log(*args):print(*args,flush=True)

def aggregate_source(name):
    with np.load(ROOT/'analysis'/f'{name}.npz') as z:
        m=z['meta'];a=z['values'];dates=pd.to_datetime(z['months'],format="%b' %Y").to_period('M')
        out={metric:defaultdict(lambda:np.zeros(a.shape[1])) for metric in ['Volume Sales(Kg)','Value Sales (1,000 Toomans)']}
        for mi,ai in zip(m,a):
            if mi[0]=='City':out[mi[13]][tuple(str(mi[c]).strip() for c in COLS)]+=np.nan_to_num(ai,nan=0)
        return out,dates

def prepare_panel():
    old,od=aggregate_source('jun_sales');new,nd=aggregate_source('aug_sales')
    dates=pd.period_range('2022-03','2026-08',freq='M')
    keys=sorted(set(old['Volume Sales(Kg)'])|set(new['Volume Sales(Kg)']))
    arrays=[];revisions={}
    for metric in old:
        a=np.zeros((len(keys),len(dates)))
        overlap=[d for d in od if d in nd]
        abs_delta=0.
        for i,k in enumerate(keys):
            ov=old[metric].get(k);nv=new[metric].get(k)
            for j,d in enumerate(dates):
                if d in nd:a[i,j]=nv[nd.get_loc(d)] if nv is not None else 0.
                elif d in od:a[i,j]=ov[od.get_loc(d)] if ov is not None else 0.
            for d in overlap:
                v1=ov[od.get_loc(d)] if ov is not None else 0.
                v2=nv[nd.get_loc(d)] if nv is not None else 0.
                abs_delta+=abs(v1-v2)
        arrays.append(a);revisions[metric]=abs_delta
    v,value=arrays
    assert np.isfinite(v).all() and (v>=0).all()
    meta=pd.DataFrame(keys,columns=DIMS)
    info={'source':'Master Mast- Aug\' 25.xlsx','source_sheets':['Jun26','Aug 26'],
          'history_start':str(dates[0]),'history_end':str(dates[-1]),'months':len(dates),
          'series':len(keys),'overlap_absolute_difference':revisions,
          'grain':DIMS,'zero_policy':'Observed zero retained; not interpreted as no demand or discontinued.',
          'taxonomy_policy':'Current source classification; no SKU row deleted; additive sales aggregated.'}
    np.savez_compressed(OUT/'prepared_panel.npz',volume=v,value=value,meta=np.array(keys),months=np.array([str(d) for d in dates]))
    return meta,v,value,dates,info

class FeatureBuilder:
    """Construct features from a prefix only. This object NEVER receives holdout actuals."""
    def __init__(self,meta,y,value,dates):
        self.meta=meta.reset_index(drop=True);self.y=y;self.dates=dates;self.cache={}
        self.price=np.divide(value,y,out=np.full_like(value,np.nan),where=y>0)
        self.parent={}
        for name,cols in [('brand_city',['City','Company','Brand']),('market_city',['City'])]:
            ids=self.meta.groupby(cols,sort=False).ngroup().to_numpy()
            totals=np.zeros((ids.max()+1,y.shape[1]));np.add.at(totals,ids,y)
            self.parent[name]=totals[ids]

    def features(self,t,h):
        assert 11<=t<self.y.shape[1] and h in [1,2,3]
        if (t,h) in self.cache:return self.cache[(t,h)]
        hist=self.y[:,:t+1];n=len(hist);latest=hist[:,-1]
        x=self.meta.copy();num={}
        for lag in [0,1,2,3,5,11,12]:
            num['volume_latest' if lag==0 else f'volume_minus_{lag}']=self.y[:,t-lag] if t>=lag else np.zeros(n)
        for w in [3,6,12]:
            z=hist[:,-w:];num[f'mean_{w}']=z.mean(1);num[f'std_{w}']=z.std(1);num[f'positive_count_{w}']=(z>0).sum(1)
        scale=np.maximum.reduce([num['mean_3'],num['mean_12']*.1,np.ones(n)])
        num['recent_growth']=np.divide(num['mean_3']-hist[:,-6:-3].mean(1),scale)
        num['last_change']=(latest-hist[:,-2])/scale
        num['positive_count_all']=(hist>0).sum(1)
        pos=hist>0;ever=pos.any(1)
        num['months_since_positive']=np.where(ever,t-np.max(np.where(pos,np.arange(t+1),-1),axis=1),t+1)
        num['months_since_first_positive']=np.where(ever,t-np.argmax(pos,axis=1),0)
        target=self.dates[t]+h
        num['target_month']=np.full(n,target.month);num['target_days']=np.full(n,calendar.monthrange(target.year,target.month)[1])
        num['month_sin']=np.full(n,np.sin(2*np.pi*target.month/12));num['month_cos']=np.full(n,np.cos(2*np.pi*target.month/12))
        num['origin_index']=np.full(n,t)
        ref=t+h-12;assert ref<=t
        num['target_month_previous_year']=self.y[:,ref]
        for name,ar in self.parent.items():
            par=ar[:,:t+1]
            num[name+'_latest']=par[:,-1]
            num[name+'_mean3']=par[:,-3:].mean(1)
            num[name+'_growth3']=np.divide(par[:,-3:].mean(1),np.maximum(par[:,-6:-3].mean(1),1))-1
            num['share_of_'+name]=latest/np.maximum(par[:,-1],1)
        ph=self.price[:,:t+1]
        for lag in [0,1,2]:num[f'price_minus_{lag}']=ph[:,t-lag]
        pcount=np.isfinite(ph[:,-3:]).sum(1)
        num['price_mean3']=np.divide(np.nansum(ph[:,-3:],axis=1),pcount,out=np.full(n,np.nan),where=pcount>0)
        num['price_change']=np.divide(ph[:,-1],ph[:,-2],out=np.full(n,np.nan),where=ph[:,-2]>0)-1
        for key,val in num.items():x[key]=np.asarray(val,dtype=float)
        x=x.replace([np.inf,-np.inf],np.nan)
        self.cache[(t,h)]=(x,scale,ever)
        return x,scale,ever

def baseline_forecasts(y,origin):
    hist=y[:,:origin+1];last=hist[:,-1];mean=hist[:,-3:].mean(1)
    forecasts={
        'Last month':np.repeat(last[:,None],3,axis=1),
        'Mean 3 months':np.repeat(mean[:,None],3,axis=1),
        'Seasonal naive':np.stack([hist[:,origin+h-12] for h in [1,2,3]],axis=1),
    }
    prev=hist[:,origin-14:origin-11].mean(1)
    growth=np.clip(np.divide(mean,prev,out=np.ones_like(mean),where=prev>0),.5,2.)
    adj=forecasts['Seasonal naive']*growth[:,None]
    adj[prev<=0,:]=mean[prev<=0,None]
    forecasts['Seasonal adjusted']=adj
    level=hist[:,0].copy()
    for t in range(1,hist.shape[1]):level=.5*hist[:,t]+.5*level
    forecasts['Exponential smoothing']=np.repeat(level[:,None],3,axis=1)
    return forecasts

def ml_forecasts(builder,origin,save=False):
    n=len(builder.meta);out={name:np.zeros((n,3)) for name in ML};audit=[];importance=[]
    signature=hashlib.sha256(json.dumps(PRESETS,sort_keys=True).encode()+builder.y[:,:origin+1].tobytes()+builder.meta.to_csv(index=False).encode()).hexdigest()[:12]
    cache_dir=OUT/'fit_cache';cache_dir.mkdir(exist_ok=True)
    for h in [1,2,3]:
        xparts=[];scales=[];ys=[];latest=[]
        for t in range(11,origin-h+1):
            x,scale,ever=builder.features(t,h)
            if not ever.any():continue
            xparts.append(x.loc[ever]);scales.append(scale[ever]);ys.append(builder.y[ever,t+h]);latest.append(builder.y[ever,t])
        xtrain=pd.concat(xparts,ignore_index=True)
        train_scale=np.concatenate(scales);train_y=np.concatenate(ys);train_latest=np.concatenate(latest)
        xp,ps,pe=builder.features(origin,h)
        for name in ML:
            start=time.monotonic();scaled='scaled' in name
            checkpoint=cache_dir/f'{origin}_{h}_{int(scaled)}_{signature}.npz'
            audit_row={'model':name,'origin':str(builder.dates[origin]),'horizon':h,'training_rows':len(xtrain),'max_training_target':str(builder.dates[origin]),'max_feature_origin':str(builder.dates[origin]-h),'prediction_input_end':str(builder.dates[origin])}
            if not save and checkpoint.exists():
                with np.load(checkpoint) as cached:out[name][:,h-1]=cached['prediction']
                audit.append(audit_row);log('REUSED',str(builder.dates[origin]),h,name);continue
            target=(train_y-train_latest)/train_scale if scaled else train_y-train_latest
            weights=np.sqrt(train_scale) if scaled else None
            model=CatBoostRegressor(**PRESETS)
            model.fit(xtrain,target,cat_features=DIMS,sample_weight=weights)
            pred=model.predict(xp)
            pred=builder.y[:,origin]+pred*(ps if scaled else 1.)
            pred=np.maximum(pred,0);pred[~pe]=0
            out[name][:,h-1]=pred
            audit.append(audit_row)
            if not save:np.savez_compressed(checkpoint,prediction=pred)
            if save:
                model.save_model(str(OUT/(('catboost_scaled' if scaled else 'catboost_residual')+f'_h{h}.cbm')))
                importance.extend({'model':name,'horizon':h,'feature':f,'importance':float(v)} for f,v in zip(xtrain.columns,model.get_feature_importance()))
            log('FIT',str(builder.dates[origin]),h,name,'rows',len(xtrain),'seconds',round(time.monotonic()-start,1))
    return out,audit,importance

LEVELS={
    'Total':[],
    'Brand':['Company','Brand'],
    'SubBrand':['Company','Brand','SubBrand'],
    'Segment':['Company','Brand','SubBrand','Subcategory','Subcategory2','Subcategory3','Subcategory4'],
    'City segment':DIMS,
}

def grouped(meta,a,cols):
    if not cols:return pd.DataFrame({'Market':['24 Cities']}),a.sum(0,keepdims=True)
    codes=meta.groupby(cols,sort=True).ngroup().to_numpy()
    vals=np.zeros((codes.max()+1,a.shape[1]));np.add.at(vals,codes,a)
    labels=meta[cols].copy();labels['_id']=codes;labels=labels.drop_duplicates('_id').sort_values('_id').drop(columns='_id').reset_index(drop=True)
    return labels,vals

def metrics(y,p):
    denom=y.sum();diff=p-y
    return {'actual':float(denom),'predicted':float(p.sum()),'absolute_error':float(np.abs(diff).sum()),'wape':float(np.abs(diff).sum()/denom) if denom else None,'bias':float(diff.sum()/denom) if denom else None,'mae':float(np.abs(diff).mean())}

def evaluate(meta,y,predictions,stage,origin):
    records=[]
    for level,cols in LEVELS.items():
        _,yy=grouped(meta,y,cols)
        for name,p in predictions.items():
            _,pp=grouped(meta,p,cols)
            for h in [0,1,2,3]:
                mm=metrics(yy,pp) if h==0 else metrics(yy[:,h-1:h],pp[:,h-1:h])
                records.append({'stage':stage,'origin':str(origin),'level':level,'model':name,'horizon':h,**mm})
    return records

def add_blends(preds,best_baseline):
    for name in ML:preds[name+' + baseline']=.5*preds[name]+.5*preds[best_baseline]

def main():
    meta,volume,value,dates,info=prepare_panel()
    cutoff=dates.get_loc(pd.Period('2026-05',freq='M'))
    # ML receives only the training prefix. Future volume/value are inaccessible to it.
    train=volume[:,:cutoff+1].copy();train_value=value[:,:cutoff+1].copy();train_dates=dates[:cutoff+1]
    builder=FeatureBuilder(meta,train,train_value,train_dates)
    origins=[dates.get_loc(pd.Period(v,freq='M')) for v in ['2025-09','2025-12','2026-02']]
    validation={};audit=[];allrecords=[]
    for origin in origins:
        preds=baseline_forecasts(train,origin)
        ml,aa,_=ml_forecasts(builder,origin);preds.update(ml);audit.extend(aa)
        validation[origin]=preds
    baseline_scores={name:0. for name in BASELINES}
    for origin,preds in validation.items():
        for name in BASELINES:
            _,yy=grouped(meta,train[:,origin+1:origin+4],LEVELS['Segment']);_,pp=grouped(meta,preds[name],LEVELS['Segment'])
            baseline_scores[name]+=np.abs(yy-pp).sum()
    best_baseline=min(baseline_scores,key=baseline_scores.get)
    scores=defaultdict(lambda:[0.,0.])
    for origin,preds in validation.items():
        add_blends(preds,best_baseline)
        yval=train[:,origin+1:origin+4]
        allrecords.extend(evaluate(meta,yval,preds,'validation',dates[origin]))
        _,yy=grouped(meta,yval,LEVELS['Segment'])
        for name,p in preds.items():
            _,pp=grouped(meta,p,LEVELS['Segment']);scores[name][0]+=np.abs(yy-pp).sum();scores[name][1]+=yy.sum()
    scores={k:v[0]/v[1] for k,v in scores.items()}
    selected=min(scores,key=scores.get)
    selection={'primary_metric':'WAPE at national brand/subbrand/full-subcategory combination level','validation_origins':[str(dates[o]) for o in origins],'validation_target_end':'2026-05','scores':scores,'best_baseline':best_baseline,'selected_model':selected,'holdout_origin':'2026-05','holdout_targets':['2026-06','2026-07','2026-08'],'presets':PRESETS,'seed':42}
    # Freeze and persist selection BEFORE forecasting or evaluating the holdout.
    (OUT/'selection_locked.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    log('SELECTION_LOCKED',json.dumps(selection))
    preds=baseline_forecasts(train,cutoff)
    ml,aa,importance=ml_forecasts(builder,cutoff,save=True);preds.update(ml);audit.extend(aa);add_blends(preds,best_baseline)
    np.savez_compressed(OUT/'predictions_locked.npz',**{k:v for k,v in preds.items()})
    # Actuals are first used below. No later changes to fitted models or selection.
    actual=volume[:,cutoff+1:cutoff+4]
    allrecords.extend(evaluate(meta,actual,preds,'holdout',dates[cutoff]))
    pd.DataFrame(allrecords).to_csv(OUT/'model_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False,encoding='utf-8-sig')
    prediction=preds[selected]
    # Full detail remains reviewable, including true zero rows and cold-start misses.
    history_last3=train[:,-3:];ever=(train>0).any(1)
    detail=meta.copy()
    detail['Positive history before June']=ever
    for j,mon in enumerate(['Mar','Apr','May']):detail[f'{mon} actual kg']=history_last3[:,j]
    for j,mon in enumerate(['Jun','Jul','Aug']):
        detail[f'{mon} actual kg']=actual[:,j];detail[f'{mon} forecast kg']=prediction[:,j];detail[f'{mon} abs error kg']=np.abs(actual[:,j]-prediction[:,j])
    detail.to_csv(OUT/'city_segment_forecasts.csv',index=False,encoding='utf-8-sig')
    workbook_detail={}
    for level in ['Brand','SubBrand','Segment']:
        labels,yy=grouped(meta,actual,LEVELS[level]);_,pp=grouped(meta,prediction,LEVELS[level]);_,base=grouped(meta,preds[best_baseline],LEVELS[level])
        rows=labels.copy()
        for j,mon in enumerate(['Jun','Jul','Aug']):
            rows[mon+' actual kg']=yy[:,j];rows[mon+' forecast kg']=pp[:,j];rows[mon+' baseline kg']=base[:,j]
        rows['Actual 3m kg']=yy.sum(1);rows['Absolute error kg']=np.abs(pp-yy).sum(1)
        rows['WAPE']=np.divide(rows['Absolute error kg'],rows['Actual 3m kg'],out=np.full(len(rows),np.nan),where=rows['Actual 3m kg']>0)
        rows=rows.sort_values('Actual 3m kg',ascending=False)
        rows.to_csv(OUT/(level.lower()+'_forecasts.csv'),index=False,encoding='utf-8-sig')
        workbook_detail[level]={'columns':list(rows.columns),'rows':rows.astype(object).where(pd.notna(rows),None).values.tolist()}
    holdout_records=[r for r in allrecords if r['stage']=='holdout']
    active=(train[:,-6:]>0).any(1)
    errors=np.abs(prediction-actual).sum(1);den=actual.sum(1)
    fractions={str(th):float(den[(errors<=th*den)&(den>0)].sum()/den.sum()) for th in [.1,.2,.3]}
    info.update({'train_end':'2026-05','test_months':['2026-06','2026-07','2026-08'],'training_months':train.shape[1],
                 'observed_series_before_test':int(ever.sum()),'active_last6_before_test':int(active.sum()),
                 'zero_history_series':int((~ever).sum()),'cold_start_actual_volume_share':float(actual[~ever].sum()/actual.sum()),
                 'actual_volume_share_with_series_error_below':fractions,
                 'limitation':'Retrospective fixed-origin test of published monthly RA estimates. Current taxonomy. No archived publication timestamps or raw-month reconstruction. Does not simulate forecasting from wall-clock May with the two-month release delay.',
                 'source_notes':'Jun26 used only for history before Dec 2023; Aug 26 for Dec 2023 onward. No Year/MAT/YTD or future prices/indices.'})
    report={'info':info,'selection':selection,'holdout_metrics':holdout_records,'validation_metrics':[r for r in allrecords if r['stage']=='validation'],'details':workbook_detail,'training_audit':audit}
    (OUT/'report_data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'training_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    # Meaningful guard checks: target dates and known-history construction at the cutoff.
    assert all(a['max_training_target']<=a['origin']<=str(dates[cutoff]) for a in audit)
    assert len(train_dates)==train.shape[1] and str(train_dates[-1])=='2026-05'
    assert all(p.shape==actual.shape and np.isfinite(p).all() and (p>=0).all() for p in preds.values())
    _,sumbrand=grouped(meta,prediction,LEVELS['Brand'])
    assert np.allclose(sumbrand.sum(0),prediction.sum(0),rtol=1e-10)
    assert builder.y.shape[1]==cutoff+1 and builder.price.shape[1]==cutoff+1
    log('FINAL',json.dumps({'selected':selected,'validation':scores,'holdout':[r for r in holdout_records if r['model']==selected and r['level'] in ['Total','Brand','SubBrand','Segment','City segment']],'info':info},ensure_ascii=False))

if __name__=='__main__':main()
