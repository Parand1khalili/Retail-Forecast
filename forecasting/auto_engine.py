"""Automatic global-model benchmark and future-only inference for RA panels.

Recipe selection uses rolling historical origins; a fixed May-2026 origin is
scored afterwards. No held-out targets enter features, fitting or model choice.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from app_core import Dataset,baseline,DIMS,ROOT
sys.path.insert(0,str(ROOT/'analysis/model_deps'))
from catboost import CatBoostRegressor
from backtest import FeatureBuilder
from light_benchmark import candidates as light_candidates
from selection_core import (rolling_plan, candidate_info, candidate_registry, choose_champion, ChampionGuard)
from category_selection import (category_dimensions, select_category_overrides,
                                forecast_with_recipes, series_recipes, holdout_comparison,
                                filtered_holdout_summary)

OUT=ROOT/'outputs/automatic_models'
OUT.mkdir(parents=True,exist_ok=True)
PARAMS=dict(iterations=180,depth=5,learning_rate=.06,l2_leaf_reg=10,
            loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,
            allow_writing_files=False,has_time=True,one_hot_max_size=32)
VERSION='global-v3-frequencies-and-market-levels'
SELECTION_VERSION='robust-price-volume-v3'
POLICY_VERSION='robust-champion-v6-rolling-audit'

def emit(*args):print(*args,flush=True)

def unit_prices(d):
    p=np.divide(d.value,d.volume,out=np.full_like(d.value,np.nan),where=d.volume>0)
    return pd.DataFrame(p).ffill(axis=1).fillna(0).to_numpy()

def price_shift_guard(d,cutoff):
    """Detect an exceptional aggregate unit-value change using past data only.

    Extrapolating a sudden level jump as continuing monthly growth was unstable.
    In that regime retain the latest sales level and use the robust median of
    preceding monthly price changes, rather than compounding the latest jump.
    This is a price/mix proxy, not an external inflation observation.
    """
    if cutoff<max(6,12//d.period_months):return {'active':False}
    volume=d.volume[:,:cutoff+1].sum(0);value=d.value[:,:cutoff+1].sum(0)
    p=np.divide(value,volume,out=np.full_like(value,np.nan),where=volume>0)
    changes=np.divide(p[1:],p[:-1],out=np.full(len(p)-1,np.nan),where=p[:-1]>0)-1
    prior=changes[:-1][-(24//d.period_months):];prior=prior[np.isfinite(prior)]
    if len(prior)<8 or not np.isfinite(changes[-1]):return {'active':False}
    center=float(np.median(prior));mad=float(np.median(abs(prior-center)))
    limit=max(.05,3*1.4826*mad)
    return {'active':bool(abs(changes[-1]-center)>limit),'current_change':float(changes[-1]),
            'median_prior_change':center,'deviation_limit':limit,
            'reference_months':len(prior)*d.period_months,'reference_periods':len(prior),
            'rule':'absolute deviation from prior-24-month median > max(5 percentage points, 3 scaled MAD)'}

def guarded_money(d,cutoff):
    guard=price_shift_guard(d,cutoff)
    if not guard['active']:return None,guard
    factor=max(.01,1+guard['median_prior_change'])
    # Preserve a newly observed price level while still forecasting quantity.
    # A fixed equal-weight blend reduces dependence on a single volume model;
    # weights are not fitted to the later monetary targets.
    models=fit_models(d,'volume_kg',cutoff,'all')
    last_volume=d.volume[:,cutoff]
    quantity=.5*models['raw']+.5*last_volume[:,None] if 'raw' in models else np.repeat(last_volume[:,None],3,axis=1)
    price=unit_prices(d)[:,cutoff]
    pred=np.stack([quantity[:,h-1]*price*factor**h for h in [1,2,3]],axis=1)
    guard['quantity_rule']='fixed 50% raw-residual global volume model + 50% last observed volume'
    return pred,guard

class Features(FeatureBuilder):
    def __init__(self,d,metric,start,end):
        self.volume=d.volume[:,start:end+1]
        self.money=d.value[:,start:end+1]
        y=self.volume if metric=='volume_kg' else self.money
        dates=pd.PeriodIndex(d.months[start:end+1],freq='M')
        super().__init__(pd.DataFrame(d.meta,columns=d.all_dimensions()),y,self.money,dates,
                         periods_per_year=12//d.period_months,period_months=d.period_months)
        self.price=np.divide(self.money,self.volume,out=np.full_like(self.money,np.nan),where=self.volume>0)
    def features(self,t,h):
        x,scale,ever=super().features(t,h)
        # Each origin/horizon is consumed once in this engine. Retaining full
        # zero-heavy feature panels would otherwise use gigabytes unnecessarily.
        self.cache.pop((t,h),None)
        x=x.rename(columns={c:c.replace('volume_','sales_') for c in x.columns})
        x=x.copy();x['horizon']=h
        for lag in sorted(set([0,1,2,5,min(11,12//self.period_months-1)])):
            x[f'physical_volume_lag{lag}']=self.volume[:,t-lag]
        x['physical_volume_mean3']=self.volume[:,t-2:t+1].mean(1)
        x['physical_volume_mean6']=self.volume[:,t-5:t+1].mean(1)
        return x,scale,ever

def fit_models(d,metric,cutoff,window,predict_only=None):
    start=max(0,cutoff-window+1) if isinstance(window,int) else 0
    if cutoff-start+1<18:return {}
    fingerprint=hashlib.sha256(VERSION.encode()+json.dumps(PARAMS,sort_keys=True).encode()+
        str((metric,start,cutoff,d.periodicity)).encode()+json.dumps(d.all_dimensions()).encode()+d.meta.tobytes()+d.volume[:,start:cutoff+1].tobytes()+d.value[:,start:cutoff+1].tobytes()).hexdigest()[:20]
    folder=OUT/'cache'/fingerprint;folder.mkdir(parents=True,exist_ok=True)
    cached=folder/'predictions.npz'
    if cached.exists():
        with np.load(cached) as z:return {k:z[k] for k in z.files}
    b=Features(d,metric,start,cutoff);last=b.y.shape[1]-1
    frames=[];truths=[];latest=[];scales=[];pt=[];pl=[];priceweights=[]
    for t in range(max(5,12//d.period_months-1),last):
        for h in [1,2,3]:
            if t+h>last:continue
            x,s,ever=b.features(t,h)
            if not ever.any():continue
            frames.append(x.loc[ever]);truths.append(b.y[ever,t+h]);latest.append(b.y[ever,t]);scales.append(s[ever])
            pt.append(b.price[ever,t+h]);pl.append(b.price[ever,t]);priceweights.append(b.volume[ever,t])
    if not frames:return {}
    X=pd.concat(frames,ignore_index=True);y=np.concatenate(truths);ly=np.concatenate(latest);scale=np.concatenate(scales)
    ptarget=np.concatenate(pt);plast=np.concatenate(pl);pw=np.concatenate(priceweights)
    outputs={};predframes=[b.features(last,h) for h in [1,2,3]]
    for mode in ['raw','scaled','log','price']:
        if mode=='price' and metric!='volume_kg':continue
        begun=time.monotonic();valid=np.ones(len(y),dtype=bool);weights=None
        if mode=='raw':response=y-ly
        elif mode=='scaled':response=(y-ly)/scale;weights=np.sqrt(scale)
        elif mode=='log':response=np.log1p(y)-np.log1p(ly);weights=np.sqrt(scale)
        else:
            valid=np.isfinite(ptarget)&np.isfinite(plast)&(ptarget>0)&(plast>0)&(pw>0)
            response=np.log(np.maximum(ptarget,1e-8)/np.maximum(plast,1e-8));weights=np.sqrt(np.maximum(pw,1))
        if not valid.any():continue
        options=dict(PARAMS)
        if np.ptp(response[valid])==0:options['allow_const_label']=True
        model=CatBoostRegressor(**options)
        model.fit(X.loc[valid],response[valid],cat_features=d.all_dimensions(),sample_weight=weights[valid] if weights is not None else None)
        model.save_model(str(folder/(mode+'.cbm')))
        arr=[]
        for h,(xp,ps,ever) in enumerate(predframes,1):
            delta=model.predict(xp)
            if mode=='raw':pred=b.y[:,last]+delta
            elif mode=='scaled':pred=b.y[:,last]+delta*ps
            elif mode=='log':pred=np.expm1(np.log1p(b.y[:,last])+np.clip(delta,-1.5,1.5))
            else:
                price=pd.DataFrame(b.price).ffill(axis=1).fillna(0).to_numpy()[:,-1]
                pred=price*np.exp(np.clip(delta,-.5,.7))
            pred=np.maximum(pred,0);pred[~ever]=0;arr.append(pred)
        outputs[mode]=np.stack(arr,axis=1)
        emit('FIT',metric,d.months[cutoff],'history',window,mode,'rows',len(X),'sec',round(time.monotonic()-begun,1))
    np.savez_compressed(cached,**outputs)
    (folder/'audit.json').write_text(json.dumps({'metric':metric,'input_start':d.months[start],'input_end':d.months[cutoff],
        'max_training_target':d.months[cutoff],'rows':len(X),'parameters':PARAMS},indent=2),encoding='utf-8')
    return outputs

def base_candidates(d,metric,cutoff):
    y=d.metric(metric);out={}
    for window in [2,3,6,12,24,'all']:
        if isinstance(window,int) and cutoff+1<window:continue
        hist=y[:,max(0,cutoff-window+1) if isinstance(window,int) else 0:cutoff+1]
        for method in ['Last month','Mean 2','Mean 3','Mean 6','Exponential smoothing','Damped trend','Seasonal naive']:
            required={'Last month':2,'Mean 2':2,'Mean 3':3,'Mean 6':6,'Damped trend':6,'Seasonal naive':12//d.period_months}
            if method in required and window!=required[method]:continue
            try:pred=np.stack([baseline(hist,method,h,12//d.period_months) for h in [1,2,3]],axis=1)
            except ValueError:continue
            out[f'base|{method}|{window}']=pred
    if metric=='value_1000_toman':
        p=unit_prices(d)[:,:cutoff+1];last=p[:,-1]
        for w in [2,3,6]:
            if p.shape[1]<w:continue
            prev=p[:,-w]
            growth=np.divide(last,prev,out=np.ones_like(last),where=prev>0)
            growth=np.clip(growth**(1/max(w-1,1)),.9,1.15)
            for damping in [0,.5,1]:
                for vm in ['Last month','Mean 2']:
                    v=baseline(d.volume[:,:cutoff+1],vm,1)
                    out[f'price-trend|{w}|{damping}|{vm}']=np.stack([v*last*growth**(h*damping) for h in [1,2,3]],axis=1)
    # Include the dependency-light benchmark in the SAME candidate grid, with
    # identical origins and forecast horizons as the CatBoost recipes.
    if cutoff>=11:
        for h in [1,2,3]:
            for name,pred in light_candidates(d.volume[:,:cutoff+1],d.value[:,:cutoff+1],metric,h,
                                              season_length=12//d.period_months).items():
                key=f'light|{name}'
                if key not in out:out[key]=np.zeros((len(d.meta),3))
                out[key][:,h-1]=pred
    return out

def candidates(d,metric,cutoff):
    out=base_candidates(d,metric,cutoff)
    for window in [24,'all']:
        fitted=fit_models(d,metric,cutoff,window)
        for mode in ['raw','scaled','log']:
            if mode in fitted:out[f'ml|{mode}|{window}']=fitted[mode]
        if metric=='value_1000_toman':
            volumes=fit_models(d,'volume_kg',cutoff,window)
            if 'price' in volumes:
                for mode in ['raw','scaled','log']:
                    out[f'price-volume|{mode}|{window}']=volumes['price']*volumes[mode]
                for vm in ['Last month','Mean 2']:
                    v=baseline(d.volume[:,:cutoff+1],vm,1)
                    out[f'price-volume|{vm}|{window}']=volumes['price']*v[:,None]
    return out

def group(y,meta,level):
    cols={'city_segment':list(range(meta.shape[1])),'city_subbrand':[0,1,2,3],'national_segment':list(range(1,8)),'total':[]}[level]
    if not cols:return y.sum(0,keepdims=True)
    labels=pd.DataFrame(meta[:,cols]);ids=labels.groupby(list(labels.columns),sort=False).ngroup().to_numpy()
    out=np.zeros((ids.max()+1,y.shape[1]));np.add.at(out,ids,y);return out

def scores(y,p):
    den=y.sum()
    return {'wape':float(abs(y-p).sum()/den) if den else None,'bias':float((p-y).sum()/den) if den else None,'actual':float(den)}

def select_recipes(d,metric,origins,by_origin,selection_mode='guarded'):
    """Compare all candidates on completed common targets at identical origins.

    Candidate training and feature construction end at each historical origin.
    Guarded selection prefers stable gains; legacy reproduces pooled-WAPE ranking.
    """
    if selection_mode not in ('guarded','legacy'):
        raise ValueError('Unknown model selection mode.')
    if not origins:raise ValueError('At least one completed validation origin is required.')
    common=set.intersection(*(set(by_origin[o]) for o in origins))
    # Exclude invalid candidate recipes entirely, rather than letting NaN win.
    common={name for name in common if all(np.isfinite(by_origin[o][name]).all() and
                                        (by_origin[o][name]>=0).all() for o in origins)}
    if not common:raise ValueError('No finite, nonnegative common candidates across validation origins.')
    policy={};validation=[];truth=d.metric(metric)
    grain='market_segment' if 'Market' in d.all_dimensions() else 'city_segment'
    for h in [1,2,3]:
        if any(o+h>=len(d.months) for o in origins):
            raise ValueError('Validation target extends beyond available observations.')
        actual=np.concatenate([group(truth[:,o+h:o+h+1],d.meta,'city_segment') for o in origins],axis=1)
        pp={name:np.concatenate([group(by_origin[o][name][:,h-1:h],d.meta,'city_segment') for o in origins],axis=1)
            for name in sorted(common)}
        simple=[name for name in pp if name.startswith(('base|','light|','price-trend|'))]
        if not simple:raise ValueError('No non-ML reference available for selection.')
        bestbase=min(simple,key=lambda name:np.abs(pp[name]-actual).sum())
        recipes={name:{'components':[[name,1.]]} for name in pp}
        for name in list(pp):
            if not name.startswith(('ml|','price-volume|')):continue
            for weight in [.25,.5,.75]:
                blend=f'blend|{name}|{weight}'
                pp[blend]=weight*pp[name]+(1-weight)*pp[bestbase]
                recipes[blend]={'components':[[name,weight],[bestbase,1-weight]]}
        candidate_scores=[]
        for name,pred in pp.items():
            row={'metric':metric,'horizon':h,'model':name,**scores(actual,pred)}
            validation.append(row)
            candidate_scores.append({'name':name,'wape':row['wape'],'bias':row['bias'],
                                     'components':recipes[name]['components']})
        # The selection denominator is identical for every candidate, so
        # minimizing city-cell absolute error is equivalent to pooled WAPE.
        ranked=sorted(candidate_scores,key=lambda row:(float('inf') if row['wape'] is None
                                                        else row['wape'],row['name']))
        losses={name:np.abs(pred-actual).sum(axis=0) for name,pred in pp.items()}
        guarded_winner,decision=choose_champion(losses,bestbase)
        winner=decision['legacy_best'] if selection_mode=='legacy' else guarded_winner
        decision={**decision,'selected':winner,'mode':selection_mode,
                  'reason':('legacy pooled WAPE winner' if selection_mode=='legacy' else decision['reason'])}
        by_month=[]
        for o in origins:
            observed=group(truth[:,o+h:o+h+1],d.meta,'city_segment')
            pieces=recipes[winner]['components']
            predicted=sum(weight*group(by_origin[o][name][:,h-1:h],d.meta,'city_segment')
                          for name,weight in pieces)
            detail=scores(observed,predicted)
            by_month.append({'origin':d.months[o],'target':d.months[o+h],
                             'actual':float(observed.sum()),'predicted':float(predicted.sum()),
                             'wape':detail['wape'],'absolute_error':float(np.abs(observed-predicted).sum())})
        policy[str(h)]={**recipes[winner],'name':winner,
                        'selection_decision':decision,
                        'validation_wape':scores(actual,pp[winner])['wape'],
                        'validation_origins':[d.months[o] for o in origins],
                        'comparison_grain':grain,
                        'comparison_metric':'pooled WAPE',
                        'selection_evidence':{
                            'metric':'pooled WAPE','grain':grain,
                            'scope':'all_series','horizon':h,'selected':winner,
                            'candidate_scores':ranked,
                            'candidate_registry':candidate_registry(pp.keys()),
                            'selection_decision':decision,
                            'origin_details':by_month
                        }}
    return policy,validation

def study():
    d=Dataset.builtin();dates=pd.PeriodIndex(d.months,freq='M')
    origins=[dates.get_loc(pd.Period(s)) for s in ['2025-05','2025-09','2026-02']]
    final=dates.get_loc(pd.Period('2026-05'))
    plan={'version':VERSION,'selection_version':POLICY_VERSION,'selection_origins':[d.months[o] for o in origins],
          'selection_target_end':'2026-05','test_origin':'2026-05','test_targets':['2026-06','2026-07','2026-08'],
          'primary_metric':'pooled city_segment WAPE','training_windows':[24,'all'],'model_parameters':PARAMS,
          'models':['CatBoost residual and price-volume','statistical baselines','lightweight benchmarks','fixed-weight blends'],
          'price_shift_guard':price_shift_guard(d,final),
          'note':'Retrospective current-taxonomy regression audit. Guard added after diagnosing an unstable price extrapolation; final dates were already examined, so this is NOT an untouched test. Prospective confirmation is still needed.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    selected={};validation=[];later=[]
    for metric in ['volume_kg','value_1000_toman']:
        vals={o:candidates(d,metric,o) for o in origins}
        selected[metric],records=select_recipes(d,metric,origins,vals)
        validation.extend(records)
        (OUT/'policy.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
        final_preds=candidates(d,metric,final)
        chosen=np.column_stack([sum(weight*final_preds[name][:,h-1] for name,weight in selected[metric][str(h)]['components']) for h in [1,2,3]])
        np.savez_compressed(OUT/(metric+'_final.npz'),prediction=chosen,actual=d.metric(metric)[:,final+1:final+4])
        for level in ['city_segment','city_subbrand','national_segment','total']:
            truth=group(d.metric(metric)[:,final+1:final+4],d.meta,level)
            pred=group(chosen,d.meta,level)
            for h in [1,2,3]:later.append({'metric':metric,'level':level,'horizon':h,**scores(truth[:,h-1:h],pred[:,h-1:h])})
        emit('SELECTED',metric,json.dumps(selected[metric]))
    pd.DataFrame(validation).to_csv(OUT/'validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(later).to_csv(OUT/'test_results.csv',index=False,encoding='utf-8-sig')
    (OUT/'results.json').write_text(json.dumps({'plan':plan,'policy':selected,'test':later},indent=2),encoding='utf-8')
    emit('FINISHED',json.dumps(later))

def forecast_component(d,metric,cutoff,name):
    # A single registry validates candidate capabilities; actual training stays
    # in the established globally fitted components and their SHA-keyed cache.
    spec=candidate_info(name)
    if metric not in spec.metrics or d.periodicity not in spec.periodicities:
        raise ValueError('Forecast candidate is incompatible with the metric or periodicity.')
    if cutoff+1 < spec.min_history:
        raise ValueError('Insufficient history for selected candidate.')
    parts=name.split('|')
    if parts[0] in ['base','light','price-trend']:return base_candidates(d,metric,cutoff)[name]
    window=int(parts[2]) if parts[2]!='all' else 'all'
    if parts[0]=='ml':return fit_models(d,metric,cutoff,window)[parts[1]]
    v=fit_models(d,'volume_kg',cutoff,window)
    vol=v[parts[1]] if parts[1] in ['raw','scaled','log'] else baseline(d.volume[:,:cutoff+1],parts[1],1)[:,None]
    return v['price']*vol

def _selection_origins(d,cutoff,strategy):
    """Deterministic, frequency-independent folds measured in observed periods."""
    plan=rolling_plan(cutoff,strategy,categories_available=bool(category_dimensions(d)))
    return plan['selection'],plan['holdout']


def dataset_policy(d,metric,cutoff,strategy='hierarchical'):
    """Select recipes exclusively on completed historical target observations."""
    if strategy not in ('hierarchical','global'):
        raise ValueError('روش انتخاب مدل باید سلسله‌مراتبی یا Global باشد.')
    configuration=rolling_plan(cutoff,strategy,categories_available=bool(category_dimensions(d)))
    digest=hashlib.sha256(VERSION.encode()+POLICY_VERSION.encode()+metric.encode()+strategy.encode()+
        json.dumps(configuration['configuration'],sort_keys=True).encode()+
        json.dumps(ChampionGuard().__dict__,sort_keys=True).encode()+
        d.periodicity.encode()+json.dumps(d.all_dimensions()).encode()+d.meta.tobytes()+
        d.volume[:,:cutoff+1].tobytes()+d.value[:,:cutoff+1].tobytes()).hexdigest()[:24]
    cached=OUT/'runtime_policies'/f'{digest}.json';cached.parent.mkdir(exist_ok=True)
    if cached.exists():return json.loads(cached.read_text(encoding='utf-8'))
    origins,holdout=configuration['selection'],configuration['holdout']
    by_origin={o:candidates(d,metric,o) for o in origins}
    mode='guarded' if holdout is not None else 'legacy'
    policy,_=select_recipes(d,metric,origins,by_origin,selection_mode=mode)
    # The previous Global policy is computed on identical selection observations,
    # without training any additional candidate model.
    legacy_policy=None
    if holdout is not None:
        legacy_policy,_=select_recipes(d,metric,origins,by_origin,selection_mode='legacy')
    policy['_legacy_global']={h:{'name':legacy_policy[h]['name'],
                                 'components':legacy_policy[h]['components']}
                               for h in ('1','2','3')} if legacy_policy is not None else None
    policy['_selection_origins']=[d.months[o] for o in origins]
    policy['_strategy']=strategy
    policy['_hierarchy']={'dimensions':category_dimensions(d),'overrides':{},
                          'groups_considered':0,'groups_selected':0}
    policy['_holdout']=None
    if holdout is not None:
        hierarchy=select_category_overrides(d,metric,origins,by_origin,policy)
        policy['_hierarchy']=hierarchy
        # Independent audit is REPORT-ONLY: never gate or tune policies on it.
        held=candidates(d,metric,holdout)
        policy['_holdout']=holdout_comparison(d,metric,holdout,held,policy,hierarchy,
                                               legacy_recipes=legacy_policy)
    policy['_validation']={'mode':mode,'selection_origins':policy['_selection_origins'],
                            'holdout_origin':d.months[holdout] if holdout is not None else None,
                            'independent':holdout is not None,
                            'periodicity':d.periodicity,
                            'period_months':d.period_months,
                            'selection_policy_version':POLICY_VERSION,
                            'plan_reason':configuration['reason'],
                            'configuration':configuration['configuration'],
                            'champion_guard':ChampionGuard().__dict__}
    for h in ('1','2','3'):
        policy[h]['basis']='انتخاب تاریخی Global و در صورت شواهد پایدار، Recipe اختصاصی گروه؛ بدون آموزش جداگانه CatBoost برای هر SKU'
    cached.write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    return policy


def automatic_forecast(d,metric,horizon,cutoff=None,strategy='hierarchical'):
    if 'Market' in d.all_dimensions() and len(set(d.meta[:,d.all_dimensions().index('Market')]))>1:
        raise ValueError('ابتدا یک سطح Market را انتخاب کنید؛ سطح‌های مکانی هم‌پوشان هستند.')
    for field in ('Element Caption','ElementCaption'):
        if field in d.all_dimensions() and len(set(d.meta[:,d.all_dimensions().index(field)]))>1:
            raise ValueError('ابتدا سطح Element Caption را انتخاب کنید؛ سطوح SKU و Product نباید با هم ترکیب شوند.')
    cutoff=len(d.months)-1 if cutoff is None else cutoff
    if strategy not in ('hierarchical','global'):
        raise ValueError('روش انتخاب مدل نامعتبر است.')
    if metric=='value_1000_toman':
        guarded,guard=guarded_money(d,cutoff)
        if guarded is not None:
            return guarded[:,horizon-1],{'name':'robust price-shift forecast',
                    'basis':'جهش غیرمعمول ارزش واحد؛ استفاده از قاعده محافظ فروش ریالی',
                    'validation_wape':None,'price_guard':guard,
                    'category_selection':{'strategy':strategy,'active':False,
                                          'reason':'Price-shift safety override'}}
    if cutoff+1<18:
        method='Mean 2' if metric=='volume_kg' and horizon==1 else (
            'Damped trend' if metric!='volume_kg' and cutoff+1>=6 else 'Last month')
        history=d.metric(metric)[:,:cutoff+1]
        return baseline(history,method,horizon),{'name':method,
                'basis':'تاریخچه برای آموزش مدل کامل کوتاه است؛ برآورد پایه',
                'validation_wape':None,
                'category_selection':{'strategy':strategy,'active':False,
                                      'reason':'Insufficient history'}}
    policy=dataset_policy(d,metric,cutoff,strategy)
    selected=policy[str(horizon)]
    hierarchy=policy['_hierarchy']
    recipes=series_recipes(d,selected,hierarchy,horizon) if strategy=='hierarchical' else [selected]*len(d.meta)
    unique={r['name']:r for r in recipes}
    # The same global candidates are used for all category branches.
    necessary={component for item in unique.values() for component,_ in item['components']}
    predictions={name:forecast_component(d,metric,cutoff,name) for name in necessary}
    output,usage,_=forecast_with_recipes(d,horizon,predictions,selected,hierarchy,strategy)
    recipe=dict(selected)
    recipe['category_selection']={
        'strategy':strategy,'active':bool(hierarchy.get('overrides',{}).get(str(horizon))) and strategy=='hierarchical',
        'dimensions':hierarchy.get('dimensions',[]),
        'groups_considered':hierarchy.get('groups_considered',0),
        'groups_selected':len(hierarchy.get('overrides',{}).get(str(horizon),[])),
        'global_recipe':selected['name'],'usage':usage,
        'selection_origins':policy['_selection_origins'],
        'holdout':(policy['_holdout'] or {}).get('horizons',{}).get(str(horizon)),
        'holdout_origin':(policy['_holdout'] or {}).get('origin'),
        'holdout_is_independent':bool(policy['_holdout']),
        'evaluation_summary':(policy['_holdout'] or {}).get('horizons',{}).get(str(horizon),{}).get('level_summary'),
        'selection_decision':selected.get('selection_decision'),
    }
    return output,recipe


def _selected_category_explanation(d,metric,horizon,mask,strategy,recipe):
    """Small, filter-specific metadata; group scores are never labeled SKU scores."""
    summary=recipe.get('category_selection')
    if not summary or not summary.get('dimensions'):
        return summary
    if strategy=='global':
        return summary
    policy=dataset_policy(d,metric,len(d.months)-1,strategy)
    rows=series_recipes(d,policy[str(horizon)],policy['_hierarchy'],horizon)
    selected={}
    for i in np.flatnonzero(mask):
        item=rows[int(i)]
        key=(item['name'],tuple(item.get('path',[])))
        if key not in selected:
            active_path=item.get('path',[])
            meaningful=[dim for dim in policy['_hierarchy'].get('dimensions',[])
                        if str(d.meta[int(i),d.all_dimensions().index(dim)]).strip().casefold()
                        not in ('','all','not specified','-', '—')]
            origin=('Parent fallback' if active_path and len(active_path)<len(meaningful)
                    else ('Category' if active_path else 'Global / inherited'))
            selected[key]={'recipe':item['name'],'path':active_path,
                           'source':origin,
                           'series':0,'group_selection_metrics':item.get('evidence',{})}
        selected[key]['series']+=1
    return {**summary,'selected_groups':sorted(selected.values(),
        key=lambda entry:(-entry['series'],entry['recipe']))[:12],
            'selected_groups_total':len(selected),
            'note':'Category metrics are measured on selection origins at category scope, not on this filtered SKU.'}

def future_request(d,request):
    if len(d.months)<2:raise ValueError('دست‌کم دو ماه کامل لازم است.')
    metric=request.get('metric','volume_kg');d.metric(metric)
    horizon=int(request.get('horizon',1))
    if not 1<=horizon<=3:raise ValueError('افق پشتیبانی‌شده یک تا سه ماه بعد از آخرین ماه فایل است.')
    filters=request.get('filters',{})
    d=d.for_market(filters.get('Market'))
    d=d.for_grain(filters.get('Element Caption') or filters.get('ElementCaption'))
    mask=d.mask(request.get('filters',{}))
    if not mask.any():raise ValueError('ترکیب انتخاب‌شده در فایل موجود نیست.')
    strategy=request.get('strategy','hierarchical')
    cells,recipe=automatic_forecast(d,metric,horizon,strategy=strategy)
    category_summary=_selected_category_explanation(d,metric,horizon,mask,strategy,recipe)
    if (category_summary and strategy=='hierarchical'
            and category_summary.get('holdout_origin')):
        # The selected series receive their own OUT-OF-SAMPLE error metrics.
        # This does not influence model selection or recipe thresholds.
        historical_policy=dataset_policy(d,metric,len(d.months)-1,strategy)
        holdout_index=d.months.index(category_summary['holdout_origin'])
        held_candidates=candidates(d,metric,holdout_index)
        category_summary['selected_filter_holdout']=filtered_holdout_summary(
            d,metric,mask,horizon,holdout_index,held_candidates,
            historical_policy[str(horizon)],historical_policy['_hierarchy'])
    target=str(pd.Period(d.months[-1],freq='M')+horizon*d.period_months)
    history=[{'month':d.months[j],'value':float(d.metric(metric)[mask,j].sum())} for j in range(max(0,len(d.months)-12),len(d.months))]
    benchmark=None;selected_benchmark=None
    report=OUT/'results.json'
    if d.periodicity=='monthly' and 'Market' not in d.all_dimensions() and report.exists() and any(word in d.source.lower() for word in ['mast','yoghurt','ماست']):
        result=json.loads(report.read_text(encoding='utf-8'))
        if result.get('plan',{}).get('selection_version')!=POLICY_VERSION:result={'test':[]}
        benchmark=next((r for r in result['test'] if r['metric']==metric and r['level']=='city_segment' and r['horizon']==horizon),None)
        ref=Dataset.builtin();reference_mask=ref.mask(request.get('filters',{}))
        if reference_mask.any() and benchmark is not None:
            with np.load(OUT/(metric+'_final.npz')) as z:
                actual=float(z['actual'][reference_mask,horizon-1].sum())
                predicted=float(z['prediction'][reference_mask,horizon-1].sum())
            selected_benchmark={'target':['2026-06','2026-07','2026-08'][horizon-1],
                                'actual':actual,'predicted':predicted,
                                'ape':abs(predicted-actual)/actual if actual>0 else None}
    return {'source':d.source,'metric':metric,'unit':d.volume_unit if metric=='volume_kg' else d.value_unit,
            'periodicity':d.periodicity,'period_months':d.period_months,
            'cutoff':d.months[-1],'target':target,'horizon':horizon,'window':len(d.months),'method':'automatic',
            'calibration':recipe,'prediction':float(cells[mask].sum()),'actual_if_known':None,
            'absolute_error_if_known':None,'rows':int(mask.sum()),'history':history,'filters':request.get('filters',{}),
            'selected_series_check':{'origins':0,'wape':None},'benchmark':benchmark,'selected_benchmark':selected_benchmark,
            'selection_evidence':recipe.get('selection_evidence'),
            'category_selection':category_summary,
            'note':('تاریخچه برای آموزش مدل کامل کوتاه است و برآورد پایه ارائه شده است. ' if len(d.months)<18 else 'روش و طول تاریخچه با آزمون ماه‌های گذشته انتخاب و مدل با داده‌های موجود آموزش داده شد. ')+
                   'دوره مقصد در فایل ورودی وجود ندارد. عدد خطای مرجع، بازآزمایی مراحل توسعه است و تأیید مستقل آینده محسوب نمی‌شود. خطای تاریخی، تضمین خطای دوره آینده یا تک‌محصول نیست.'}

if __name__=='__main__':study()
