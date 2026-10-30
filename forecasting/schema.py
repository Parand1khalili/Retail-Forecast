"""Conservative RA spreadsheet schema detection.

Ambiguous fields are sent back to the UI for explicit mapping. Sales units are
kept as written in the source; no currency or mass conversion is guessed.
"""
from __future__ import annotations
import re, unicodedata
from difflib import SequenceMatcher

FIELDS=['Market','City','Company','Brand','SubBrand','Subcategory','Subcategory2','Subcategory3','Subcategory4','Measure']
REQUIRED=['City','Brand','Measure']
ALIASES={
    'Market':['Market','Market Type','Market Level','نوع بازار','سطح بازار'],
    'City':['City/Region/Dis. Channel','City/Region/Dis.Channel','City','City Name','Town','شهر','نام شهر','شهر منطقه کانال'],
    'Company':['Company','Company Name','Manufacturer','Producer','کمپانی','شرکت','تولیدکننده'],
    'Brand':['Brand','Brand Name','نام برند','برند'],
    'SubBrand':['SubBrand','Sub Brand','Sub-brand','Subbrand Name','زیر برند','زیر‌برند','ساب برند'],
    'Subcategory':['Subcategory','Subcategory 1','Sub Category','Sub Category 1','Category','Category 1','ساب کتگوری','ساب‌کتگوری','دسته فرعی','زیر دسته'],
    'Subcategory2':['Subcategory 2','Sub Category 2','Subcategory2','ساب کتگوری 2','زیر دسته 2'],
    'Subcategory3':['Subcategory 3','Sub Category 3','Subcategory3','ساب کتگوری 3','زیر دسته 3'],
    'Subcategory4':['Subcategory 4','Sub Category 4','Subcategory4','ساب کتگوری 4','زیر دسته 4'],
    'Measure':['Measure','Metric','Measure Type','Sales Measure','Sales Metric','Indicator','شاخص','شاخص فروش','نوع شاخص','سنجه'],
}
MONTHS={'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
        'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,
        'september':9,'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12}
PERSIAN_DIGITS=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')

def normalized(value):
    s=unicodedata.normalize('NFKC',str(value or '')).translate(PERSIAN_DIGITS).lower()
    s=s.replace('ي','ی').replace('ك','ک')
    return ''.join(c for c in s if c.isalnum())

def month(value):
    s=unicodedata.normalize('NFKC',str(value or '')).translate(PERSIAN_DIGITS).strip().lower()
    m=re.fullmatch(r"([a-z]+)\s*['/\-\s]\s*(\d{2}|\d{4})",s)
    if m and m.group(1) in MONTHS:
        year=int(m.group(2));year=2000+year if year<100 else year
        return f'{year:04d}-{MONTHS[m.group(1)]:02d}' if 2000<=year<=2099 else None
    m=re.fullmatch(r'(20\d{2})[-/](0?[1-9]|1[0-2])',s)
    if m:return f'{int(m.group(1)):04d}-{int(m.group(2)):02d}'
    m=re.fullmatch(r'(0?[1-9]|1[0-2])[-/](20\d{2})',s)
    if m:return f'{int(m.group(2)):04d}-{int(m.group(1)):02d}'
    return None

def months_in_header(head):
    found={i:month(v) for i,v in head.items()}
    return {i:v for i,v in found.items() if v is not None}

def resolve_fields(head,manual=None):
    manual=manual or {}
    occurrences={}
    for i,value in head.items():
        if str(value).strip():occurrences.setdefault(normalized(value),[]).append(i)
    # Duplicate normalized labels cannot be mapped by name without ambiguity.
    byname={label:indices[0] for label,indices in occurrences.items() if len(indices)==1}
    used=set();result={};confidence={}
    for field in FIELDS:
        requested=str(manual.get(field,'')).strip()
        if requested and requested!='*':
            idx=byname.get(normalized(requested))
            if idx is not None and idx not in used:
                result[field]=idx;used.add(idx);confidence[field]='manual';continue
            # An explicit mapping must never silently fall back to a guess.
            continue
        hits=[byname[normalized(a)] for a in ALIASES[field] if normalized(a) in byname and byname[normalized(a)] not in used]
        if len(set(hits))==1:
            idx=hits[0];result[field]=idx;used.add(idx);confidence[field]='alias'
    # High confidence typo correction, only when unique and substantially better
    # than the second closest field. Never take a weak Brand/SubBrand guess.
    for field in FIELDS:
        if field in result:continue
        candidates=[]
        for label,idx in byname.items():
            if idx in used:continue
            score=max(SequenceMatcher(None,label,normalized(a)).ratio() for a in ALIASES[field])
            if score>=.91:candidates.append((score,idx))
        candidates.sort(reverse=True)
        if candidates and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>=.06):
            _,idx=candidates[0];result[field]=idx;used.add(idx);confidence[field]='near match'
    return result,confidence

def metric_kind(value,manual=None):
    raw=str(value or '').strip()
    if manual:
        for kind,label in manual.items():
            if label and normalized(raw)==normalized(label):return kind
    s=normalized(raw)
    if any(x in s for x in ['share','percent','درصد','سهم']) or '%' in raw:return None
    sales=('sales' in s or 'فروش' in s or 'turnover' in s)
    if not sales:return None
    if any(x in s for x in ['volume','quantity','qty','kilogram','kg','liter','litre','unit','count','حجم','مقدار','تعداد']):return 'volume_kg'
    if any(x in s for x in ['value','amount','revenue','tooman','toman','rial','ارزش','مبلغ','ریال','تومان']):return 'value_1000_toman'
    return None

def metric_unit(kind,label,override=None):
    if override and str(override).strip():return str(override).strip()
    s=normalized(label)
    if kind=='volume_kg':
        if any(x in s for x in ['kilogram','kg','کیلوگرم']):return 'kg'
        if any(x in s for x in ['liter','litre','لیتر']) or re.search(r'\(\s*(l|lit|lt)\s*\)',str(label),re.I):return 'لیتر'
        if any(x in s for x in ['unit','count','تعداد']):return 'عدد'
        return 'واحد حجم فایل'
    if any(x in s for x in ['1000tooman','1000toman','هزارتومان']):return 'هزار تومان'
    if any(x in s for x in ['milliontooman','میلیونتومان']):return 'میلیون تومان'
    if any(x in s for x in ['1000rial','هزارریال']):return 'هزار ریال'
    if any(x in s for x in ['millionrial','میلیونریال']):return 'میلیون ریال'
    if any(x in s for x in ['tooman','toman','تومان']):return 'تومان'
    if any(x in s for x in ['rial','ریال']):return 'ریال'
    return 'واحد مبلغ فایل'

def mapping_prompt(sheet,head,detected,examples=None):
    return {'mapping_required':True,'sheet':sheet,
            'headers':[str(v) for _,v in sorted(head.items()) if str(v).strip() and month(v) is None],
            'suggested':{field:str(head[idx]) for field,idx in detected.items()},
            'required':REQUIRED,'fields':FIELDS,
            'measure_examples':list(examples or [])[:12],
            'message':'سرتیترهای این فایل به‌طور مطمئن شناسایی نشدند. ستون‌ها و نام دو شاخص فروش را یک بار تعیین کنید.'}

class SchemaError(ValueError):
    def __init__(self,prompt):
        self.prompt=prompt
        super().__init__(prompt['message'])
