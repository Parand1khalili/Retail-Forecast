"""Local RA report reader and cutoff-safe forecasting service.

No network services are used. Monthly columns are published RA estimates; all
calculations below use the City market only, so geographic views never overlap.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib, json, re, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from schema import REQUIRED, SchemaError, mapping_prompt, metric_kind, metric_unit, months_in_header, normalized, resolve_fields

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'outputs/forecast_app/cache'
CACHE.mkdir(parents=True, exist_ok=True)
DIMS = ['City','Company','Brand','SubBrand','Subcategory','Subcategory2','Subcategory3','Subcategory4']
METRICS = {'volume_kg':'Volume Sales(Kg)', 'value_1000_toman':'Value Sales (1,000 Toomans)'}
M='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
AVAILABLE_METHODS=['Last month','Mean 2','Mean 3','Mean 6','Exponential smoothing','Damped trend','Seasonal naive']

def colnum(ref):
    n=0
    for c in ref:
        if c.isdigit():break
        n=n*26+ord(c.upper())-64
    return n-1

def read_sheet(z, shared, xml):
    """Yield dictionaries per row without loading a full 100+ MB workbook."""
    with z.open(xml) as f:
        for _,el in ET.iterparse(f,events=('end',)):
            if el.tag!=M+'row':continue
            vals={}
            for c in el:
                v=c.find(M+'v');value=v.text if v is not None else None
                if c.attrib.get('t')=='s' and value is not None:value=shared[int(value)]
                elif c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.iter(M+'t'))
                if value is not None:vals[colnum(c.attrib['r'])]=value
            el.clear()
            yield vals

def parse_master(path,mapping=None):
    mapping=mapping or {}
    manual_fields=mapping.get('fields',{})
    manual_measures=mapping.get('measure_values',{})
    manual_units=mapping.get('units',{})
    sheets=[];pending_prompt=None;units={}
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                for _,el in ET.iterparse(f,events=('end',)):
                    if el.tag==M+'si':
                        shared.append(''.join(t.text or '' for t in el.iter(M+'t')));el.clear()
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets={r.attrib['Id']:r.attrib['Target'] for r in rels}
        for sh in wb.find(M+'sheets'):
            t=targets[sh.attrib[R+'id']].replace('\\','/')
            xml=t.lstrip('/') if t.startswith('/') else 'xl/'+t
            rows=read_sheet(z,shared,xml)
            head=None;months={}
            for _ in range(10):
                try:candidate=next(rows)
                except StopIteration:break
                maybe=months_in_header(candidate)
                if len(maybe)>=2:head,months=candidate,maybe;break
            if head is None:continue
            if len(set(months.values()))!=len(months):
                raise ValueError(f'در شیت {sh.attrib["name"]} چند ستون برای یک ماه وجود دارد؛ ستون درست را مشخص کنید.')
            fields,_=resolve_fields(head,manual_fields)
            if any(field not in fields for field in REQUIRED):
                if pending_prompt is None:pending_prompt=mapping_prompt(sh.attrib['name'],head,fields)
                continue
            ordered=sorted(months)
            bykey=defaultdict(lambda:np.zeros((2,len(ordered)),dtype=float))
            metric_keys=[set(),set()]
            valid=np.zeros(len(ordered),dtype=int);missing=np.zeros(len(ordered),dtype=int)
            seen_measures=set();labels={}
            for vals in rows:
                if 'Market' in fields and normalized(vals.get(fields['Market'])) not in ['city','cities','شهر','شهرها']:
                    continue
                metric=vals.get(fields['Measure'])
                if metric is not None and len(seen_measures)<12:seen_measures.add(str(metric))
                kind=metric_kind(metric,manual_measures)
                if kind is None:continue
                metric_index=0 if kind=='volume_kg' else 1
                labels[kind]=str(metric)
                key=tuple(str(vals.get(fields[dim],'')).strip() if dim in fields else 'Not specified' for dim in DIMS)
                if not key[0] or not key[2]:continue
                metric_keys[metric_index].add(key)
                a=bykey[key][metric_index]
                for j,col in enumerate(ordered):
                    raw=vals.get(col)
                    try:
                        num=float(raw)
                        if not np.isfinite(num) or num<0:raise ValueError()
                        a[j]+=num;valid[j]+=1
                    except (TypeError,ValueError):missing[j]+=1
            if not metric_keys[0] and not metric_keys[1]:
                if pending_prompt is None:pending_prompt=mapping_prompt(sh.attrib['name'],head,fields,sorted(seen_measures))
                continue
            if metric_keys[0]!=metric_keys[1]:
                raise SchemaError(mapping_prompt(sh.attrib['name'],head,fields,sorted(seen_measures)))
            partial=[months[ordered[j]] for j in range(len(ordered)) if valid[j]>0 and missing[j]>0]
            if partial:raise ValueError(f'در شیت {sh.attrib["name"]} برای ماه‌های {", ".join(partial)} سلول فروش خالی است؛ قبل از پیش‌بینی بررسی شود.')
            keep=[j for j in range(len(ordered)) if valid[j]>0]
            if not keep:continue
            for kind,label in labels.items():
                unit=metric_unit(kind,label,manual_units.get(kind))
                if kind in units and units[kind]!=unit:
                    raise ValueError(f'واحد شاخص {kind} بین شیت‌ها فرق دارد؛ آن‌ها را بدون تبدیل نمی‌توان ترکیب کرد.')
                units[kind]=unit
            dates=[months[ordered[j]] for j in keep]
            sheets.append((dates,{k:v[:,keep] for k,v in bykey.items()},sh.attrib['name']))
    if not sheets:
        if pending_prompt:raise SchemaError(pending_prompt)
        raise ValueError('هیچ شیت فروش ماهانه پیدا نشد. فایل باید دست‌کم دو ستون ماهانه و ردیف‌های حجم و مبلغ فروش داشته باشد.')
    # Latest report sheet is authoritative for overlapping monthly columns.
    sheets.sort(key=lambda s:max(s[0]))
    keys=sorted(set().union(*(set(s[1]) for s in sheets)))
    dates=sorted(set().union(*(set(s[0]) for s in sheets)))
    if len(dates)<2:raise ValueError('دست‌کم دو ماه فروشِ کامل برای پیش‌بینی لازم است.')
    periods=pd.PeriodIndex(dates,freq='M')
    if not all(periods[i]+1==periods[i+1] for i in range(len(periods)-1)):
        raise ValueError('ماه‌های فروش در فایل پیوسته نیستند؛ لطفاً یک مستر با ستون‌های ماهانهٔ کامل بدهید.')
    a=np.zeros((2,len(keys),len(dates)),dtype=float)
    index={k:i for i,k in enumerate(keys)};di={d:i for i,d in enumerate(dates)}
    for months,series,_ in sheets:
        jj=[di[d] for d in months]
        # Replace whole sheet-month (including groups that disappeared) rather
        # than combining overlapping report snapshots.
        a[:,:,jj]=0
        for key,val in series.items():a[:,index[key],jj]=val
    return Dataset(np.array(keys,dtype=str),a[0],a[1],dates,Path(path).name,units['volume_kg'],units['value_1000_toman'])

@dataclass
class Dataset:
    meta:np.ndarray
    volume:np.ndarray
    value:np.ndarray
    months:list[str]
    source:str
    volume_unit:str='kg'
    value_unit:str='هزار تومان'

    @classmethod
    def builtin(cls):
        with np.load(ROOT/'outputs/yoghurt_backtest/prepared_panel.npz') as z:
            return cls(z['meta'],z['volume'],z['value'],z['months'].astype(str).tolist(),'نمونهٔ ماست، مستر اوت ۲۰۲۶')

    @classmethod
    def file(cls,path,mapping=None):
        path=Path(path)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
        signature=hashlib.sha256(json.dumps(mapping or {},sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()[:12]
        out=CACHE/('v2_'+digest.hexdigest()+'_'+signature+'.npz')
        if out.exists():
            with np.load(out) as z:
                return cls(z['meta'],z['volume'],z['value'],z['months'].astype(str).tolist(),path.name,str(z['volume_unit']),str(z['value_unit']))
        obj=parse_master(path,mapping)
        np.savez_compressed(out,meta=obj.meta,volume=obj.volume,value=obj.value,months=np.array(obj.months),volume_unit=obj.volume_unit,value_unit=obj.value_unit)
        return obj

    def metric(self,key):
        if key=='volume_kg':return self.volume
        if key=='value_1000_toman':return self.value
        raise ValueError('شاخص فروش معتبر نیست.')

    def mask(self,filters):
        mask=np.ones(len(self.meta),dtype=bool)
        for j,dim in enumerate(DIMS):
            value=filters.get(dim)
            if value and value!='*':mask &= self.meta[:,j]==value
        return mask

    def options(self):
        return {'source':self.source,'months':self.months,'dimensions':DIMS,
                'rows':self.meta.tolist(),'series_count':len(self.meta),
                'metric_labels':METRICS,'units':{'volume_kg':self.volume_unit,'value_1000_toman':self.value_unit}}

def baseline(hist,method,horizon):
    if hist.ndim==1:hist=hist[None,:]
    if method=='Last month':return hist[:,-1]
    if method.startswith('Mean '):
        n=int(method.rsplit(' ',1)[1])
        if hist.shape[1]<n:raise ValueError(f'برای میانگین {n} ماه، دست‌کم {n} ماه داده لازم است.')
        return hist[:,-n:].mean(axis=1)
    if method=='Exponential smoothing':
        level=hist[:,0].copy()
        for j in range(1,hist.shape[1]):level=.5*hist[:,j]+.5*level
        return level
    if method=='Damped trend':
        if hist.shape[1]<6:raise ValueError('برای روند، شش ماه داده لازم است.')
        recent=hist[:,-3:].mean(axis=1);previous=hist[:,-6:-3].mean(axis=1)
        last=hist[:,-1];delta=np.clip((recent-previous)*horizon/3,-.15*last*horizon,.15*last*horizon)
        return np.maximum(last+delta,0)
    if method=='Seasonal naive':
        if hist.shape[1]<12:raise ValueError('برای الگوی فصل مشابه، دوازده ماه داده لازم است.')
        return hist[:,-12+horizon-1]
    raise ValueError('روش پیش‌بینی معتبر نیست.')

def choose_method(dataset,metric,cutoff,window,horizon):
    """Choose using only completed targets before cutoff, if enough exist.

    Evaluation at city segment grain prevents a large national total from
    concealing wrong brand/city allocations.
    """
    record=ROOT/'outputs/yoghurt_history_study/selection_locked.json'
    selection=json.loads(record.read_text(encoding='utf-8')) if record.is_file() else {}
    yogurt=any(word in dataset.source.lower() for word in ['mast','yoghurt','ماست'])
    default=selection.get(metric,{}).get(str(horizon),{}).get('method') if yogurt else None
    if default is None:default='Mean 2' if metric=='volume_kg' and horizon==1 else 'Last month'
    candidates=[m for m in AVAILABLE_METHODS if m=='Last month' or
                (m.startswith('Mean ') and window>=int(m.rsplit(' ',1)[1])) or
                (m=='Exponential smoothing' and window>=2) or
                (m=='Damped trend' and window>=6) or
                (m=='Seasonal naive' and window>=12)]
    if default not in candidates:default='Last month' if horizon>1 else ('Mean 2' if window>=2 and metric=='volume_kg' else 'Last month')
    y=dataset.metric(metric)
    # One aggregate per exact city/taxonomy combination already exists.
    origins=list(range(window-1,cutoff-horizon+1))
    origins=origins[-12:]
    if len(origins)<5:return default,{'basis':('مطالعهٔ تاریخی ماست' if yogurt else 'قاعدهٔ اولیه برای دادهٔ کوتاه')+'؛ در فایل فعلی آزمون کافی وجود ندارد.','origins':len(origins)}
    errors={m:0. for m in candidates};actual=0.
    for o in origins:
        hist=y[:,o-window+1:o+1]
        truth=y[:,o+horizon]
        actual+=truth.sum()
        for method in candidates:
            try:pred=baseline(hist,method,horizon)
            except ValueError:continue
            errors[method]+=np.abs(pred-truth).sum()
    scored={m:e/actual for m,e in errors.items() if actual>0}
    if not scored:return default,{'basis':'مطالعهٔ تاریخی ماست','origins':len(origins)}
    chosen=min(scored,key=scored.get)
    return chosen,{'basis':'آزمون ماه‌های قدیمی‌تر در همین داده تا تاریخ شروع پیش‌بینی','origins':len(origins),'wape':scored[chosen], 'alternatives':scored}

def predict(dataset,request):
    metric=request.get('metric','volume_kg');y=dataset.metric(metric)
    cutoff=request.get('cutoff',dataset.months[-1]);target=request.get('target')
    if cutoff not in dataset.months:raise ValueError('ماه پایان داده معتبر نیست.')
    origin=dataset.months.index(cutoff)
    if origin<1:raise ValueError('دست‌کم دو ماه ورودی لازم است.')
    horizon=int(request.get('horizon',1))
    if not 1<=horizon<=3:raise ValueError('افق پیش‌بینی باید بین یک تا سه ماه باشد.')
    expected=str(pd.Period(cutoff,freq='M')+horizon)
    if target and target!=expected:raise ValueError('ماه مقصد با افق انتخاب‌شده سازگار نیست.')
    window=request.get('window','all')
    if window=='all':window=origin+1
    else:window=int(window)
    if window<2 or window>origin+1:raise ValueError('طول تاریخچه معتبر نیست.')
    filters=request.get('filters',{})
    mask=dataset.mask(filters)
    if not mask.any():raise ValueError('برای ترکیب انتخاب‌شده هیچ ردیفی در مستر وجود ندارد.')
    method=request.get('method','auto')
    calibration={}
    if method=='auto':method,calibration=choose_method(dataset,metric,origin,window,horizon)
    if method not in AVAILABLE_METHODS:raise ValueError('روش پیش‌بینی معتبر نیست.')
    hist=y[mask,max(0,origin-window+1):origin+1]
    # Forecast exact city/taxonomy cells, then add; this preserves the same
    # evaluation grain used in the historical study.
    prediction=float(baseline(hist,method,horizon).sum())
    actual=float(y[mask,origin+horizon].sum()) if origin+horizon<len(dataset.months) else None
    history=[{'month':dataset.months[j],'value':float(y[mask,j].sum())} for j in range(max(0,origin-window+1),origin+1)]
    sub=y[mask]
    review_origins=list(range(window-1,origin-horizon+1))[-12:]
    review_error=0.;review_actual=0.;review_count=0
    for o in review_origins:
        truth=float(sub[:,o+horizon].sum())
        guess=float(baseline(sub[:,o-window+1:o+1],method,horizon).sum())
        review_error+=abs(guess-truth);review_actual+=truth;review_count+=1
    selection_wape=review_error/review_actual if review_actual>0 else None
    return {'source':dataset.source,'metric':metric,'unit':dataset.volume_unit if metric=='volume_kg' else dataset.value_unit,
            'cutoff':cutoff,'target':expected,'horizon':horizon,'window':window,'method':method,
            'calibration':calibration,'prediction':prediction,'actual_if_known':actual,
            'absolute_error_if_known':abs(prediction-actual) if actual is not None else None,
            'rows':int(mask.sum()),'history':history,'filters':filters,
            'selected_series_check':{'origins':review_count,'wape':selection_wape},
            'note':'ماه‌ها برآورد گزارش RA هستند؛ بازه‌های برداشت دوماهه روی هم می‌افتند. پیش‌بینی فقط از ستون‌های تا ماه پایان داده استفاده کرده است.'}
