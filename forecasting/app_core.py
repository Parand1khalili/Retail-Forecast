"""Local RA report reader and cutoff-safe forecasting service.

No network services are used. Source observations are monthly or two-monthly;
geographic levels stay separate to avoid counting overlapping aggregates.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib, json, re, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from schema import REQUIRED, SchemaError, mapping_prompt, metric_kind, metric_unit, months_in_header, normalized, resolve_fields, detect_layout

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

def read_sheet(z, shared, xml, with_row_number=False):
    """Yield dictionaries per row without loading a full 100+ MB workbook."""
    with z.open(xml) as f:
        for _,el in ET.iterparse(f,events=('end',)):
            if el.tag!=M+'row':continue
            vals={}
            for c in el:
                v=c.find(M+'v');value=v.text if v is not None else None
                if c.attrib.get('t')=='s' and value is not None:value=shared[int(value)]
                elif c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.iter(M+'t'))
                if c.attrib.get('t') in ('b','e'):value='[invalid Excel cell: '+str(value)+']'
                if value is not None:
                    idx=colnum(c.attrib['r'])
                    if idx in vals:raise ValueError('سلول تکراری در یک ردیف Excel وجود دارد.')
                    vals[idx]=value
            row_number=el.attrib.get('r','?')
            el.clear()
            yield (row_number,vals) if with_row_number else vals

def parse_master(path,mapping=None):
    """Fail closed on ambiguous sales observations; audit every accepted period."""
    mapping=mapping or {}
    periodicity=mapping.get('periodicity','monthly')
    if periodicity not in ('monthly','bimonthly'):raise ValueError('نوع دوره باید ماهانه یا دوماهه باشد.')
    period_months=1 if periodicity=='monthly' else 2
    opposite='bimonthly' if periodicity=='monthly' else 'monthly'
    manual_fields=mapping.get('fields',{})
    manual_measures=mapping.get('measure_values',{})
    manual_units=mapping.get('units',{})
    if any(k not in METRICS for k in (*manual_measures,*manual_units)):
        raise ValueError('نگاشت شاخص فروش نامعتبر است.')
    requested_measures=[normalized(v) for v in manual_measures.values() if str(v).strip()]
    if len(set(requested_measures))!=len(requested_measures):
        raise ValueError('یک نام شاخص نمی‌تواند هم حجم و هم مبلغ باشد.')
    sheets=[];units={};preview=None;layout=None;skipped=[];sheet_audits=[]
    grain_labels={'sku','skus','product','products','company','companies','brand','subbrand',
                  'sub brand','category','subcategory','sub category','total','all products','all'}
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
            name=sh.attrib['name']
            t=targets[sh.attrib[R+'id']].replace('\\','/')
            xml=t.lstrip('/') if t.startswith('/') else 'xl/'+t
            rows=read_sheet(z,shared,xml,with_row_number=True)
            head=None;possible=False;wrong_frequency=False
            for _ in range(10):
                try:_,candidate=next(rows)
                except StopIteration:break
                maybe=months_in_header(candidate,periodicity)
                alternative=months_in_header(candidate,opposite)
                guessed,_=resolve_fields(candidate)
                possible |= bool(maybe or alternative or 'Measure' in guessed)
                wrong_frequency |= len(alternative)>=2
                if len(maybe)>=2:head=candidate;break
            if head is None:
                # Scan the remaining rows for a delayed/unsupported sales table.
                # Non-sales worksheets are recorded, never silently discarded.
                for _,candidate in rows:
                    if (months_in_header(candidate,periodicity) or months_in_header(candidate,opposite)
                            or any(metric_kind(v) for v in candidate.values())):
                        possible=True;break
                if wrong_frequency:raise ValueError(f'شیت {name}: نوع دوره انتخاب‌شده با ستون‌ها سازگار نیست.')
                if possible:raise ValueError(f'شیت {name}: ساختار یا دوره‌های جدول فروش قابل تشخیص نیست؛ سرتیتر باید در ده ردیف اول باشد.')
                skipped.append({'sheet':name,'reason':'no_recognised_sales_table'})
                continue
            fields,_,categories,months,ignored=detect_layout(head,mapping)
            def prompt(message,examples=None):
                info=mapping_prompt(name,head,fields,examples,categories,months)
                info['message']=f'شیت {name}: {message}'
                return SchemaError(info)
            if (any(field not in fields for field in REQUIRED)
                or any(field not in fields for field,v in manual_fields.items() if str(v).strip() not in ('','*'))
                or any(fields[field] not in categories for field in ['City','Brand'] if field in fields)
                or len(months)<2):
                raise prompt('نگاشت ستون‌های اجباری یا دستی معتبر نیست؛ ساختار را تأیید کنید.')
            boundary=fields['Measure']
            before=[i for i in sorted(head) if i<boundary and str(head[i]).strip()]
            if any(i>=boundary for f,i in fields.items() if f!='Measure'):
                raise prompt('ستون‌های دسته‌بندی باید قبل از شاخص فروش باشند.')
            all_names=[normalized(head[i]) for i in before]
            if len(set(all_names))!=len(all_names):
                raise ValueError(f'شیت {name}: نام ستون‌های دسته‌بندی تکراری است.')
            if any(i>boundary for i in months_in_header(head,opposite)):
                raise ValueError(f'شیت {name}: نوع دوره ماهانه و دوماهه در یک جدول مخلوط است.')
            if len(set(months.values()))!=len(months):
                raise ValueError(f'شیت {name}: چند ستون برای یک دوره وجود دارد.')
            selected=mapping.get('dimension_columns')
            if selected is not None and not set(map(normalized,selected)).issubset(set(all_names)):
                raise prompt('ستون دسته‌بندی انتخاب‌شده در فایل موجود نیست.')
            grain_col=next((i for i in before if normalized(head[i])=='elementcaption'),None)
            if grain_col is not None and grain_col not in categories:
                raise prompt('Element Caption برای کنترل سطح تجمیع نباید حذف شود.')
            # Series identity is not full-row identity. Distinct source rows may
            # legitimately contribute to one series; exact duplicate rows are rejected.
            # All pre-measure dimensions are retained for pairing before projection.
            field_by_col={i:f for f,i in fields.items()}
            raw_cols=sorted(before,key=lambda i:normalized(field_by_col.get(i,head[i])))
            raw_ids=tuple(normalized(field_by_col.get(i,head[i])) for i in raw_cols)
            if len(set(raw_ids))!=len(raw_ids):
                raise prompt('دو ستون به یک هویت دسته‌بندی نگاشت شده‌اند.')
            extras=([('Market',fields['Market'])] if 'Market' in fields else [])+[
                ('Element Caption' if normalized(head[i])=='elementcaption' else str(head[i]),i)
                for i in categories if i not in fields.values()]
            extras.sort(key=lambda item:normalized(item[0]))
            dims=DIMS+[label for label,_ in extras]
            if len(set(map(normalized,dims)))!=len(dims):
                raise prompt('نام ابعاد خروجی تکراری است.')
            projection=[fields.get(dim) if fields.get(dim) in categories else None for dim in DIMS]+[i for _,i in extras]
            projected_ids=tuple(normalized(d) for d in dims)
            current_layout=(raw_ids,projected_ids,tuple(i is not None for i in projection))
            if layout is None:
                layout=current_layout;output_dims=dims
            elif layout!=current_layout:
                raise ValueError(f'شیت {name}: ابعاد جدول با شیت‌های فروش دیگر یکسان نیست؛ ترکیب خودکار امن نیست.')
            shown=[d for d,i in zip(dims,projection) if i is not None]
            ordered=sorted(months,key=lambda i:months[i])
            dates=[months[i] for i in ordered]
            bykey={};projected={};seen=set();seen_measures=set();labels={};excluded=defaultdict(int)
            row_fingerprints={};identity_counts=defaultdict(int);source_sales_rows=0
            collision_examples=[]
            raw_totals=defaultdict(lambda:np.zeros((2,len(ordered))))
            valid=np.zeros(len(ordered),dtype=int);missing=np.zeros(len(ordered),dtype=int)
            scope_by_key={}
            for row_number,vals in rows:
                metric=vals.get(boundary)
                if metric is not None:seen_measures.add(str(metric))
                auto_kind=metric_kind(metric)
                kind=metric_kind(metric,manual_measures)
                if kind is None:
                    if not str(metric or '').strip() and any(str(vals.get(i,'')).strip() for i in ordered):
                        raise ValueError(f'شیت {name}: ردیف دارای مقدار دوره بدون نام شاخص است.')
                    if str(metric or '').strip():
                        s=normalized(metric)
                        # Unknown sales-like indicators require explicit mapping;
                        # shares/distribution/other metrics are not volume/value.
                        if ('sales' in s or 'فروش' in s or 'turnover' in s) and not any(x in s for x in ('share','percent','درصد','سهم')) and '%' not in str(metric):
                            if not all(manual_measures.get(k) for k in METRICS):
                                raise prompt('شاخص فروش ناشناخته است؛ نام دقیق حجم و مبلغ را تعیین کنید.',sorted(seen_measures))
                        excluded[str(metric)]+=1
                    continue
                if kind not in METRICS:raise prompt('نوع شاخص فروش نامعتبر است.')
                if auto_kind is not None and auto_kind!=kind:
                    raise prompt('نگاشت دستی حجم و مبلغ با نوع شاخص شناخته‌شده تعارض دارد.',sorted(seen_measures))
                mi=0 if kind=='volume_kg' else 1
                unit=metric_unit(kind,metric)
                unknown=unit in ('واحد حجم فایل','واحد مبلغ فایل')
                override=manual_units.get(kind)
                if override and str(override).strip():
                    declared=metric_unit(kind,str(override))
                    if declared in ('واحد حجم فایل','واحد مبلغ فایل'):declared=str(override).strip()
                    if not unknown and declared!=unit:
                        raise ValueError(f'شیت {name}: واحد دستی شاخص {metric} با واحد صریح فایل سازگار نیست.')
                    unit=declared;unknown=False
                if unknown:raise prompt('واحد شاخص فروش نامشخص است؛ واحد واقعی فایل را تعیین کنید.',sorted(seen_measures))
                if kind in units and units[kind]!=unit:
                    raise ValueError(f'شیت {name}: واحد شاخص {kind} متفاوت است؛ ترکیب بدون تبدیل مجاز نیست.')
                units[kind]=unit
                if kind in labels and normalized(labels[kind])!=normalized(metric):
                    raise prompt('چند نام متفاوت برای یک شاخص فروش وجود دارد؛ جدول یا نام شاخص را بررسی کنید.',sorted(seen_measures))
                labels[kind]=str(metric)
                for f in ('City','Brand','Market'):
                    if f in fields and not str(vals.get(fields[f],'')).strip():
                        raise ValueError(f'شیت {name}: مقدار بعد اجباری {f} در ردیف فروش خالی است.')
                raw_key=tuple(str(vals.get(i,'')).strip() for i in raw_cols)
                # Compare all populated source cells, not just categorical keys.
                # Include non-period columns because distinct records can share
                # all forecast periods. No raw sales values enter the audit examples.
                fingerprint=hashlib.sha256(json.dumps(sorted((i,v) for i,v in vals.items() if str(v).strip()),ensure_ascii=False,separators=(',',':')).encode('utf-8')).digest()
                if fingerprint in row_fingerprints:
                    raise ValueError(f'شیت {name}: ردیف‌های Excel {row_fingerprints[fingerprint]} و {row_number} کاملاً تکراری هستند؛ جمع خودکار مجاز نیست.')
                row_fingerprints[fingerprint]=row_number
                identity=(raw_key,mi)
                identity_counts[identity]+=1
                source_sales_rows+=1
                if identity in seen and len(collision_examples)<20:
                    collision_examples.append({'row':row_number,'measure':str(metric),'occurrence':identity_counts[identity]})
                seen.add(identity)
                out_key=tuple(str(vals.get(i,'')).strip() if i is not None else 'Not specified' for i in projection)
                projected[raw_key]=out_key
                market=str(vals.get(fields['Market'],'')).strip() if 'Market' in fields else ''
                caption=str(vals.get(grain_col,'')).strip() if grain_col is not None else ''
                scope=(market,caption if caption.casefold() in grain_labels else '')
                scope_by_key[raw_key]=scope
                if raw_key not in bykey:bykey[raw_key]=np.zeros((2,len(ordered)))
                row_values=np.zeros(len(ordered),dtype=float)
                for j,col in enumerate(ordered):
                    raw=vals.get(col)
                    try:
                        num=float(raw)
                        if not np.isfinite(num) or num<0:raise ValueError()
                        row_values[j]=num;valid[j]+=1
                    except (TypeError,ValueError):
                        if raw is not None and str(raw).strip() not in ('','-','–','—'):
                            raise ValueError(f'شیت {name}: مقدار فروش نامعتبر در دوره {months[col]}.')
                        missing[j]+=1
                bykey[raw_key][mi]+=row_values
                raw_totals[scope][mi]+=row_values
            for kind,requested in manual_measures.items():
                if str(requested).strip() and normalized(labels.get(kind))!=normalized(requested):
                    raise prompt('نام صریح شاخص انتخاب‌شده در جدول فروش یافت نشد.',sorted(seen_measures))
            if not bykey:
                raise prompt('هیچ ردیف حجم و مبلغ قابل تشخیص نیست.',sorted(seen_measures))
            if any((k,0) not in seen or (k,1) not in seen for k in bykey):
                raise prompt('ردیف‌های حجم و مبلغ در هویت کامل منبع جفت نیستند.',sorted(seen_measures))
            if any((valid>0)&(missing>0)):
                bad=[date for date,v,m in zip(dates,valid,missing) if v and m]
                raise ValueError(f'شیت {name}: بخشی از سلول‌های فروش دوره‌های {", ".join(bad)} خالی یا مفقود است؛ داده مفقود صفر نیست.')
            keep=np.flatnonzero(valid>0)
            if not len(keep):
                raise ValueError(f'شیت {name}: هیچ دوره فروش با مقدار عددی موجود نیست.')
            unavailable=[date for date,v in zip(dates,valid) if not v]
            kept_dates=[dates[j] for j in keep]
            # Build independent source totals before projecting selected dimensions.
            source_groups={}
            for scope in raw_totals:
                totals=raw_totals[scope][:,keep]
                if not np.isfinite(totals).all():raise ValueError(f'شیت {name}: مجموع فروش خارج از محدوده عددی است.')
                source_groups[scope]=totals
            group_keys=defaultdict(list)
            for key,scope in scope_by_key.items():group_keys[scope].append(key)
            for scope,keys in group_keys.items():
                keys.sort()
                if not np.allclose(sum((bykey[k][:,keep] for k in keys),np.zeros_like(source_groups[scope])),
                                   source_groups[scope],rtol=1e-10,atol=0):
                    raise ValueError(f'شیت {name}: مجموع ردیف‌ها با مجموع منبع تطابق ندارد.')
                sheets.append({'name':name,'scope':scope,'dates':kept_dates,'keys':keys,
                               'values':np.stack([bykey[k][:,keep] for k in keys]),
                               'projected':[projected[k] for k in keys],'source_totals':source_groups[scope],
                               'unavailable_periods':unavailable})
            sheet_audits.append({'sheet':name,'source_series':len(bykey),'source_sales_rows':source_sales_rows,
                                 'aggregated_distinct_rows':source_sales_rows-len(seen),
                                 'multirow_measure_identities':sum(c>1 for c in identity_counts.values()),
                                 'aggregation_examples':collision_examples,
                                 'excluded_measures':dict(excluded),'ignored_columns':ignored,
                                 'unavailable_periods':unavailable,'accepted_periods':kept_dates,
                                 'excluded_dimensions':[str(head[i]) for i in before if i not in categories and i!=fields.get('Market')],
                                 'source_totals':[{'market':s[0],'grain':s[1],'periods':kept_dates,
                                                  'volume':v[0].tolist(),'value':v[1].tolist()}
                                                  for s,v in source_groups.items()]})
            if preview is None:
                preview=mapping_prompt(name,head,fields,sorted(seen_measures),categories,months)
                preview.update(mapping_required=False,periodicity=periodicity,shown_dimensions=shown)
    if not sheets:
        raise ValueError('هیچ شیت فروش با دوره‌های انتخاب‌شده پیدا نشد.')
    dates=sorted({date for sheet in sheets for date in sheet['dates']})
    if len(dates)<2:raise ValueError('دست‌کم دو دوره فروش کامل برای پیش‌بینی لازم است.')
    periods=pd.PeriodIndex(dates,freq='M')
    if not all(periods[i]+period_months==periods[i+1] for i in range(len(periods)-1)):
        raise ValueError('دوره‌های فروش پیوسته نیستند؛ ستون‌های گمشده یا نوع دوره را بررسی کنید.')
    # Complementary sheets can contain different rows at the same reporting
    # grain. Merge at full source identity x period, BEFORE projection.
    scopes=defaultdict(list)
    for sheet in sheets:scopes[sheet['scope']].append(sheet)
    output_keys=sorted({k for sheet in sheets for k in sheet['projected']})
    out_index={k:i for i,k in enumerate(output_keys)}
    a=np.zeros((2,len(output_keys),len(dates)),dtype=float)
    date_index={date:j for j,date in enumerate(dates)}
    reconciled=[];duplicates=[];projected_owners={}
    for scope,parts in sorted(scopes.items()):
        covered={date for part in parts for date in part['dates']}
        absent=sorted(set(dates)-covered)
        if absent:
            raise ValueError(f'پوشش دوره‌ها برای Market/سطح تجمیع {scope} ناقص است: {", ".join(absent)}؛ نبود داده صفر نیست.')
        projection={}
        for part in parts:
            for key,target in zip(part['keys'],part['projected']):
                if key in projection and projection[key]!=target:
                    raise ValueError('نگاشت هویت یکسان بین شیت‌ها ناسازگار است.')
                projection[key]=target
        source_keys=sorted(projection)
        raw_index={key:i for i,key in enumerate(source_keys)}
        n=len(source_keys)
        idx=np.array([out_index[projection[k]] for k in source_keys],dtype=int)
        unique=np.unique(idx)
        for i in unique:
            if i in projected_owners and projected_owners[i]!=scope:
                raise ValueError('نگاشت ابعاد باعث ترکیب سطوح Market/Element Caption شده است.')
            projected_owners[i]=scope
        merged=np.zeros((n,2,len(dates)),dtype=float)
        # Owner index distinguishes explicitly recorded zero from absent rows.
        owner=np.full((n,len(dates)),-1,dtype=np.int32)
        expected=np.zeros((2,len(dates)))
        source_rows=np.zeros((2,len(dates)))
        duplicate_totals=np.zeros((2,len(dates)))
        part_indices=[]
        for p,part in enumerate(parts):
            rows=np.array([raw_index[k] for k in part['keys']],dtype=int)
            part_indices.append(rows)
            for k,date in enumerate(part['dates']):
                j=date_index[date]
                values=part['values'][:,:,k]
                previous=owner[rows,j]
                repeated=previous>=0
                fresh=~repeated
                source_rows[:,j]+=part['source_totals'][:,k]
                if repeated.any():
                    overlap_rows=rows[repeated]
                    differs=np.any(merged[overlap_rows,:,j]!=values[repeated],axis=1)
                    if differs.any():
                        old=parts[int(previous[repeated][np.flatnonzero(differs)[0]])]['name']
                        raise ValueError(f'شیت‌های {old} و {part["name"]}: داده متعارض برای یک هویت فروش در دوره مشترک {date}؛ منبع معتبر را مشخص کنید.')
                    duplicate_totals[:,j]+=values[repeated].sum(axis=0)
                    for prior in np.unique(previous[repeated]):
                        count=int(np.sum(previous==prior))
                        duplicates.append({'period':date,'market':scope[0],'grain':scope[1],
                                           'retained_sheet':parts[int(prior)]['name'],
                                           'duplicate_sheet':part['name'],'series_count':count})
                merged[rows[fresh],:,j]=values[fresh]
                owner[rows[fresh],j]=p
                expected[:,j]+=values[fresh].sum(axis=0)
        # A genuinely missing cell/column is not a suppressed SKU. A completely
        # unavailable source period can only be accepted within the final
        # calendar if every affected identity is supplied by another sheet.
        for part,rows in zip(parts,part_indices):
            for date in part['unavailable_periods']:
                if date in date_index and np.any(owner[rows,date_index[date]]<0):
                    raise ValueError(f'شیت {part["name"]}: داده صریحاً مفقود در دوره {date}؛ نبود مقدار را نمی‌توان صفر فرض کرد.')
        inferred=(owner<0).sum(axis=0)
        if not np.allclose(source_rows,expected+duplicate_totals,rtol=1e-10,atol=0):
            raise ValueError('مجموع خام شیت‌ها با مجموع پذیرفته‌شده و تکرارهای حذف‌شده سازگار نیست.')
        for j in range(len(dates)):
            np.add.at(a[0,:,j],idx,merged[:,0,j])
            np.add.at(a[1,:,j],idx,merged[:,1,j])
        actual=a[:,unique,:].sum(axis=1)
        if not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=1e-10,atol=0):
            raise ValueError('مجموع فروش Import با مجموع معتبر Master در Market/سطح تجمیع تطابق ندارد.')
        reconciled.append({'market':scope[0],'grain':scope[1],'periods':dates,
                           'source_series':n,'imported_series':len(unique),
                           'raw_sheet_volume':source_rows[0].tolist(),'raw_sheet_value':source_rows[1].tolist(),
                           'duplicate_volume':duplicate_totals[0].tolist(),'duplicate_value':duplicate_totals[1].tolist(),
                           'source_volume':expected[0].tolist(),'source_value':expected[1].tolist(),
                           'imported_volume':actual[0].tolist(),'imported_value':actual[1].tolist(),
                           'inferred_zero_series_by_period':inferred.tolist(),
                           'inferred_zero_series_periods':int(inferred.sum()),
                           'max_absolute_difference':np.max(abs(expected-actual),axis=1).tolist()})
    preview['integrity']={'status':'passed','policy':'complementary_sheets_v2','sheets':sheet_audits,
                          'absent_identity_policy':'assume_zero_in_covered_scope_period',
                          'inferred_zero_series_periods':sum(r['inferred_zero_series_periods'] for r in reconciled),
                          'skipped_sheets':skipped,'identical_overlaps':duplicates,
                          'reconciliation':reconciled,'unit_labels':dict(units)}
    result=Dataset(np.array(output_keys,dtype=str),a[0],a[1],dates,Path(path).name,
                   units['volume_kg'],units['value_1000_toman'],output_dims,preview,periodicity)
    # Validate ambiguity even before the user attempts a forecast.
    result.report_grain_levels()
    return result

@dataclass
class Dataset:
    meta:np.ndarray
    volume:np.ndarray
    value:np.ndarray
    months:list[str]
    source:str
    volume_unit:str='kg'
    value_unit:str='هزار تومان'
    dimension_names:list[str]|None=None
    schema:dict|None=None
    periodicity:str='monthly'

    @property
    def period_months(self):
        return 2 if self.periodicity=='bimonthly' else 1

    def for_market(self,market=None):
        """Select exactly one Market level; overlapping totals must not mix."""
        if 'Market' not in self.all_dimensions():return self
        col=self.all_dimensions().index('Market')
        levels=set(self.meta[:,col])
        if not market or market=='*':
            if len(levels)>1:
                raise ValueError('ابتدا سطح بازار (Market) را انتخاب کنید تا سطوح مکانی با هم جمع نشوند.')
            return self
        selected=self.meta[:,col]==market
        if not selected.any():raise ValueError('سطح بازار انتخاب‌شده موجود نیست.')
        return Dataset(self.meta[selected],self.volume[selected],self.value[selected],
                       self.months,self.source,self.volume_unit,self.value_unit,
                       self.dimension_names,self.schema,self.periodicity)

    def report_grain_levels(self):
        """Identify genuine aggregate/SKU levels, not arbitrary Element captions.

        Element Caption can contain brand/product labels rather than grain
        indicators. Treating every distinct caption as a reporting level
        incorrectly prevents forecasts and fragments global model training.
        Ambiguous mixes are rejected instead of silently double-counted.
        """
        field=next((name for name in ('Element Caption','ElementCaption')
                    if name in self.all_dimensions()),None)
        if field is None:return [None]
        col=self.all_dimensions().index(field)
        levels=list(dict.fromkeys(str(value).strip() for value in self.meta[:,col]))
        if len(levels)<=1:return [None]
        report={'sku','skus','product','products','company','companies','brand',
                'subbrand','sub brand','category','subcategory','sub category',
                'total','all products','all'}
        recognised=[name for name in levels if name.casefold() in report]
        if not recognised:return [None]
        if len(recognised)!=len(levels):
            raise ValueError('Element Caption شامل ترکیبی از سطوح تجمیع و نام محصولات است؛ لطفاً ساختار فایل را بررسی کنید.')
        return levels

    def for_grain(self, grain=None):
        """Separate only confirmed reporting grains such as SKU and Product.

        Ordinary product labels under Element Caption remain available for
        filtering, but never force separate model training.
        """
        levels=self.report_grain_levels()
        if levels==[None]:return self
        if not grain or grain=='*':
            raise ValueError('این فایل چند سطح تجمیع محصول دارد؛ ابتدا Element Caption را انتخاب کنید تا SKU و Product با هم جمع نشوند.')
        dim=next(x for x in ('Element Caption','ElementCaption')
                 if x in self.all_dimensions())
        col=self.all_dimensions().index(dim)
        selected=self.meta[:,col]==grain
        if not selected.any():
            raise ValueError('سطح تجمیع محصول انتخاب‌شده در فایل موجود نیست.')
        return Dataset(self.meta[selected],self.volume[selected],self.value[selected],
                       self.months,self.source,self.volume_unit,self.value_unit,
                       self.dimension_names,self.schema,self.periodicity)

    def all_dimensions(self):
        return self.dimension_names or DIMS

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
        out=CACHE/('v8_'+digest.hexdigest()+'_'+signature+'.npz')
        if out.exists():
            with np.load(out) as z:
                schema=json.loads(str(z['schema_json'])) if 'schema_json' in z.files else None
                dims=z['dimension_names'].astype(str).tolist() if 'dimension_names' in z.files else DIMS
                return cls(z['meta'],z['volume'],z['value'],z['months'].astype(str).tolist(),
                           path.name,str(z['volume_unit']),str(z['value_unit']),dims,schema,
                           str(z['periodicity']) if 'periodicity' in z.files else 'monthly')
        obj=parse_master(path,mapping)
        np.savez_compressed(out,meta=obj.meta,volume=obj.volume,value=obj.value,
                            months=np.array(obj.months),volume_unit=obj.volume_unit,
                            value_unit=obj.value_unit,dimension_names=np.array(obj.all_dimensions()),
                            schema_json=json.dumps(obj.schema,ensure_ascii=False),periodicity=obj.periodicity)
        return obj

    def metric(self,key):
        if key=='volume_kg':return self.volume
        if key=='value_1000_toman':return self.value
        raise ValueError('شاخص فروش معتبر نیست.')

    def mask(self,filters):
        mask=np.ones(len(self.meta),dtype=bool)
        for j,dim in enumerate(self.all_dimensions()):
            value=filters.get(dim)
            if value and value!='*':mask &= self.meta[:,j]==value
        return mask

    def options(self):
        dims=self.all_dimensions()
        visible=(self.schema or {}).get('shown_dimensions') or dims
        if 'Market' in visible:visible=['Market']+[name for name in visible if name!='Market']
        idx=[dims.index(d) for d in visible]
        return {'source':self.source,'months':self.months,'dimensions':visible,
                'rows':self.meta[:,idx].tolist(),'series_count':len(self.meta),
                'schema':self.schema,'periodicity':self.periodicity,'period_months':self.period_months,
                'metric_labels':METRICS,'units':{'volume_kg':self.volume_unit,'value_1000_toman':self.value_unit}}

def baseline(hist,method,horizon,season_length=12):
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
        if hist.shape[1]<season_length:raise ValueError('برای مقایسه فصلی، یک سال داده لازم است.')
        return hist[:,-season_length+horizon-1]
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
            try:pred=baseline(hist,method,horizon,12//dataset.period_months)
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
    expected=str(pd.Period(cutoff,freq='M')+horizon*dataset.period_months)
    if target and target!=expected:raise ValueError('ماه مقصد با افق انتخاب‌شده سازگار نیست.')
    window=request.get('window','all')
    if window=='all':window=origin+1
    else:window=int(window)
    if window<2 or window>origin+1:raise ValueError('طول تاریخچه معتبر نیست.')
    filters=request.get('filters',{})
    dataset=dataset.for_market(filters.get('Market')).for_grain(filters.get('Element Caption') or filters.get('ElementCaption'))
    y=dataset.metric(metric)
    mask=dataset.mask(filters)
    if not mask.any():raise ValueError('برای ترکیب انتخاب‌شده هیچ ردیفی در مستر وجود ندارد.')
    method=request.get('method','auto')
    calibration={}
    if method=='auto':method,calibration=choose_method(dataset,metric,origin,window,horizon)
    if method not in AVAILABLE_METHODS:raise ValueError('روش پیش‌بینی معتبر نیست.')
    hist=y[mask,max(0,origin-window+1):origin+1]
    # Forecast exact city/taxonomy cells, then add; this preserves the same
    # evaluation grain used in the historical study.
    prediction=float(baseline(hist,method,horizon,12//dataset.period_months).sum())
    actual=float(y[mask,origin+horizon].sum()) if origin+horizon<len(dataset.months) else None
    history=[{'month':dataset.months[j],'value':float(y[mask,j].sum())} for j in range(max(0,origin-window+1),origin+1)]
    sub=y[mask]
    review_origins=list(range(window-1,origin-horizon+1))[-12:]
    review_error=0.;review_actual=0.;review_count=0
    for o in review_origins:
        truth=float(sub[:,o+horizon].sum())
        guess=float(baseline(sub[:,o-window+1:o+1],method,horizon,12//dataset.period_months).sum())
        review_error+=abs(guess-truth);review_actual+=truth;review_count+=1
    selection_wape=review_error/review_actual if review_actual>0 else None
    return {'source':dataset.source,'metric':metric,'unit':dataset.volume_unit if metric=='volume_kg' else dataset.value_unit,
            'cutoff':cutoff,'target':expected,'horizon':horizon,'window':window,'method':method,
            'calibration':calibration,'prediction':prediction,'actual_if_known':actual,
            'absolute_error_if_known':abs(prediction-actual) if actual is not None else None,
            'rows':int(mask.sum()),'history':history,'filters':filters,
            'selected_series_check':{'origins':review_count,'wape':selection_wape},
            'note':'ماه‌ها برآورد گزارش RA هستند؛ بازه‌های برداشت دوماهه روی هم می‌افتند. پیش‌بینی فقط از ستون‌های تا ماه پایان داده استفاده کرده است.'}
