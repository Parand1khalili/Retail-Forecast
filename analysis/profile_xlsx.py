from pathlib import Path
import zipfile, xml.etree.ElementTree as ET, json, re
from collections import Counter, defaultdict
import numpy as np,sys

ROOT=Path("D:/Aug'26/Aug'26")
OUT=Path(__file__).parent
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
M='{'+NS['m']+'}'

def colnum(ref):
    n=0
    for a in ref:
        if a.isdigit(): break
        n=n*26+ord(a)-64
    return n-1

def profile(p,sheet,target):
    struct=json.loads((OUT/(p.stem+'_structure.json')).read_text(encoding='utf-8'))
    sh=next(s for s in struct['sheets'] if s['name']==sheet)
    with zipfile.ZipFile(p) as z:
        ss=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                for _,el in ET.iterparse(f,events=('end',)):
                    if el.tag==M+'si':
                        ss.append(''.join(t.text or '' for t in el.iter(M+'t')));el.clear()
        metrics=Counter(); axes=[Counter() for _ in range(14)]
        rows=[]; meta=[]; source_rows=[]; headers=None; monthly=[]; measures=defaultdict(list)
        lastrow=0
        with z.open(sh['xml']) as f:
            for _,el in ET.iterparse(f,events=('end',)):
                if el.tag!=M+'row': continue
                vals={}
                for c in el:
                    v=c.find(M+'v');value=v.text if v is not None else None
                    if c.attrib.get('t')=='s' and value is not None: value=ss[int(value)]
                    elif c.attrib.get('t')=='inlineStr': value=''.join(t.text or '' for t in c.iter(M+'t'))
                    if value is not None: vals[colnum(c.attrib['r'])]=value
                rownum=int(el.attrib['r']);el.clear()
                if rownum==1:
                    headers=[vals.get(i) for i in range(max(vals)+1)]
                    monthly=[i for i,h in enumerate(headers) if h and re.match(r"^\s*[A-Za-z]{3}'\s*\d{4}$",h)]
                    continue
                if 13 not in vals: continue
                lastrow=rownum
                metric=vals[13];metrics[metric]+=1
                for i in range(14): axes[i][vals.get(i)]+=1
                arr=[]
                for i in monthly:
                    try: arr.append(float(vals.get(i)))
                    except (ValueError,TypeError): arr.append(float('nan'))
                measures[metric].append(arr[-1] if arr else float('nan'))
                if metric in ['Volume Sales(Kg)','Value Sales (1,000 Toomans)']:
                    rows.append(arr);meta.append([vals.get(i) for i in range(14)]);source_rows.append(rownum)
        arr=np.array(rows,dtype=float)
        np.savez_compressed(OUT/(target+'.npz'),values=arr,meta=np.array(meta,dtype=str),months=np.array([headers[i] for i in monthly]),source_rows=np.array(source_rows))
        result={'file':p.name,'sheet':sheet,'last_data_row':lastrow,'rows':sum(metrics.values()),'monthly_count':len(monthly),'months':[headers[i] for i in monthly],'metrics':dict(metrics),'axes':{headers[i]:dict(axes[i]) for i in [0,1,2,4,5,6,7,8,9,10,11,12]},'latest_metric_summary':{k:{'rows':len(v),'missing':int(np.isnan(v).sum()),'zero':int((np.array(v)==0).sum()),'min':float(np.nanmin(v)),'max':float(np.nanmax(v))} for k,v in measures.items()}}
        for metric in ['Volume Sales(Kg)','Value Sales (1,000 Toomans)']:
            if not meta: continue
            mask=np.array([m[13]==metric for m in meta],dtype=bool)
            a=arr[mask]; mt=np.array(meta,dtype=str)[mask]
            if not len(a): continue
            cnt=np.sum(a>0,axis=1)
            keys=[tuple(m[:13]) for m in mt]
            result[metric]={'series':len(a),'duplicate_keys':len(keys)-len(set(keys)),'unique_sku_descriptions':len(set(mt[:,3])),'all_zero_or_missing':int((cnt==0).sum()),'positive_latest':int((a[:,-1]>0).sum()),'positive_months_quantiles':np.quantile(cnt,[0,.25,.5,.75,1]).tolist(),'missing_cells':int(np.isnan(a).sum()),'negative_cells':int((a<0).sum()),'monthly_zero_fraction':np.mean(a==0,axis=0).tolist(),'monthly_missing_fraction':np.mean(np.isnan(a),axis=0).tolist(),'locations':{loc:{'series':int((mt[:,1]==loc).sum()),'latest_volume_or_value':float(np.nansum(a[mt[:,1]==loc,-1]))} for loc in sorted(set(mt[:,1]))}}
        (OUT/(target+'_profile.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('COMPLETE',target, json.dumps({k:result[k] for k in ['rows','monthly_count','metrics']},ensure_ascii=False),flush=True)

master=ROOT/"Master Mast- Aug' 25.xlsx"
jobs=[(master,'Aug 26','aug_sales'),(master,'Jun26','jun_sales'),(master,'dis aug','aug_distribution'),(ROOT/"Yoghurt- Pivot Table with sub brand -Aug'26.xlsx",'Master Data','pivot_sales')]
for job in jobs:
    if len(sys.argv)==1 or job[2] in sys.argv[1:]: profile(*job)
