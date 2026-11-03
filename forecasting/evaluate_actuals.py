"""Audit actual-vs-forecast data using a third Excel source sheet.

Run from repository root::
    python forecasting/evaluate_actuals.py path/to/retail_forecast_full.xlsx --output path/to/review

This tool never changes the workbook. Do not commit report outputs or source data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile

X = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
META = ('Market','City','Element Caption','Element Description','Subcategory',
        'Subcategory2','SKUSize','SKUPackage','Brand','Company')
ORIGINAL = ('Market','City/Region/Dis. Channel','Element Caption','Element Description',
            'Subcategory','Subcategory 2','SKUSize','SKUPackage','Brand','Company')
GROUPS = ('City','Company','Brand','Subcategory','Subcategory2','SKUSize','SKUPackage')


def norm(value):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(value or '')).strip()).casefold()


def _cells(row):
    result = {}
    for cell in row:
        ref = cell.get('r', '')
        index = 0
        for ch in ref:
            if not ch.isalpha(): break
            index = 26 * index + (ord(ch.upper()) - 64)
        if index == 0: continue
        val = cell.find(X + 'v')
        value = val.text if val is not None else None
        if cell.get('t') == 'inlineStr':
            value = ''.join(t.text or '' for t in cell.iter(X + 't'))
        result[index - 1] = value
    return result


def _shared_strings(archive):
    if 'xl/sharedStrings.xml' not in archive.namelist(): return []
    items = []
    with archive.open('xl/sharedStrings.xml') as stream:
        for _, element in ET.iterparse(stream, events=('end',)):
            if element.tag == X + 'si':
                items.append(''.join(t.text or '' for t in element.iter(X + 't')))
                element.clear()
    return items


def _sheet_rows(archive, number, strings):
    name = f'xl/worksheets/sheet{number}.xml'
    with archive.open(name) as stream:
        for _, element in ET.iterparse(stream, events=('end',)):
            if element.tag != X + 'row': continue
            cells = _cells(element)
            for child in element:
                if child.get('t') == 's':
                    col_ref = re.match(r'[A-Z]+',child.get('r',''),re.I)
                    idx = None
                    if col_ref:
                        idx = 0
                        for letter in col_ref.group(): idx = idx*26 + ord(letter.upper()) - 64
                        idx -= 1
                    # Address-aware mapping also handles physically blank columns.
                    if idx is not None and cells[idx] is not None:
                        cells[idx] = strings[int(cells[idx])]
            element.clear()
            yield cells


def _column(n):
    name = ''
    while n:
        n, digit = divmod(n - 1, 26)
        name = chr(65 + digit) + name
    return name


def _table(rows):
    it = iter(rows)
    header = next(it)
    names = {v: i for i, v in header.items() if v is not None}
    for row in it:
        yield {name: row.get(index) for name, index in names.items()}


def _number(value):
    if value is None or str(value).strip() in ('', '-', '–', '—'): return None
    result = float(value)
    if not math.isfinite(result): raise ValueError('Invalid non-finite sales number')
    return result


def _metric(text):
    name = norm(text)
    if 'volume' in name or 'حجم' in name: return 'Volume'
    if 'value' in name or 'مبلغ' in name: return 'Value'
    return None


def _key(row, headings):
    return tuple(norm(row.get(name)) for name in headings)


def _score(pairs):
    if not pairs: return {'count':0}
    n = len(pairs)
    actual = sum(a for a, _ in pairs)
    forecast = sum(p for _, p in pairs)
    absolute = sum(abs(p - a) for a, p in pairs)
    squared = sum((p - a)**2 for a, p in pairs)
    return {'count':n,'actual':actual,'forecast':forecast,
            'bias_pct':100*(forecast-actual)/actual if actual else None,
            'wape_pct':100*absolute/actual if actual else None,
            'mae':absolute/n,'rmse':math.sqrt(squared/n),
            'actual_zeros':sum(a==0 for a,p in pairs),
            'false_positive':sum(a==0 and p>0 for a,p in pairs),
            'missed_positive':sum(a>0 and p==0 for a,p in pairs)}


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_file, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecast = {}
    raw = defaultdict(lambda:defaultdict(list))
    with ZipFile(input_file) as archive:
        strings = _shared_strings(archive)
        for num, metric in ((1,'Volume'),(2,'Value')):
            for r in _table(_sheet_rows(archive,num,strings)):
                key = _key(r, META)
                if any(not v for v in key):
                    raise ValueError(f'Empty metadata key in {metric} sheet')
                point = (metric,key)
                if point in forecast: raise ValueError('Duplicate forecast key: '+repr(point))
                history = {c.removeprefix('Actual | '):_number(v) for c,v in r.items()
                           if c.startswith('Actual | ')}
                future = {c.removeprefix('Forecast | '):_number(v) for c,v in r.items()
                          if c.startswith('Forecast | ')}
                forecast[point] = (history,future)
        # Third worksheet must contain the independently published Actual values.
        for r in _table(_sheet_rows(archive,3,strings)):
            metric = _metric(r.get('Measure'))
            if metric:
                raw[(metric,_key(r, ORIGINAL))][metric].append(r)
    target_set = {tuple(d[1].keys()) for d in forecast.values()}
    if len(target_set)!=1 or not target_set:
        raise ValueError('Inconsistent future headers between Volume and Value sheets')
    target = next(iter(target_set))[0]
    matched = set(forecast) & set(raw)
    qa = {'forecast_keys':len(forecast)//2, 'actual_keys':len(raw)//2,
          'matched_measure_rows':len(matched),'unmatched_forecast':len(set(forecast)-set(raw)),
          'unmatched_actual':len(set(raw)-set(forecast)),
          'historical_compared':0,'historical_differences':0,'historical_max_abs_error':0,
          'target':target,'missing_target_rows':0}
    details = []
    for (metric,key) in sorted(matched):
        history,future = forecast[(metric,key)]
        observed_rows = raw[(metric,key)][metric]
        if not observed_rows:continue
        for period, old in history.items():
            vals = [_number(row.get(period)) for row in observed_rows]
            if old is None or any(v is None for v in vals):continue
            err = abs(old - sum(vals))
            qa['historical_compared'] += 1
            qa['historical_max_abs_error']=max(qa['historical_max_abs_error'],err)
            if err > max(1e-6, 1e-8*abs(sum(vals))):qa['historical_differences']+=1
        actuals = [_number(row.get(target)) for row in observed_rows]
        actual = sum(actuals) if all(v is not None for v in actuals) else None
        pred = future.get(target)
        if actual is None or pred is None:
            qa['missing_target_rows']+=1
            continue
        values=[v for v in history.values() if v is not None]
        positives=[v for v in values if v>0]
        adi=len(values)/len(positives) if positives else None
        mean_pos=sum(positives)/len(positives) if positives else None
        cv2=sum((v-mean_pos)**2 for v in positives)/len(positives)/(mean_pos**2) if mean_pos else None
        if not positives: demand='No historical demand'
        elif len(positives)<2: demand='Insufficient positive history'
        elif adi>1.32: demand='Lumpy' if cv2>=.49 else 'Intermittent'
        else: demand='Erratic' if cv2>=.49 else 'Smooth'
        season_length=6 if '-' in target else 12
        record = {'Metric':metric, **dict(zip(META,key)), 'Actual':actual, 'Forecast':pred,
                  'Absolute error':abs(pred-actual),
                  'Signed error':pred-actual,
                  'APE %':100*abs(pred-actual)/actual if actual>0 else None,
                  'Last period':list(history.values())[-1],
                  'Seasonal previous year':list(history.values())[-season_length] if len(history)>=season_length else None,
                  'Demand class':demand,'ADI':adi,'CV2':cv2,
                  'Positive history periods':len(positives)}
        details.append(record)
    if not details:raise ValueError('No matching real target rows')
    scores=[];shares=[];baselines=[];demand_scores=[];mat_shares=[];time_series=[]
    levels=sorted(set(r['Market'] for r in details))
    for metric in ('Volume','Value'):
        for level in levels:
            subset=[r for r in details if r['Metric']==metric and r['Market']==level]
            if not subset:continue
            pairs=[(r['Actual'],r['Forecast']) for r in subset]
            scores.append({'Metric':metric,'Market':level,**_score(pairs)})
            demand_groups=defaultdict(list)
            for row in subset: demand_groups[row['Demand class']].append(row)
            for label, subset_class in demand_groups.items():
                demand_scores.append({'Metric':metric,'Market':level,'Demand class':label,
                                      **_score([(r['Actual'],r['Forecast']) for r in subset_class])})
            # Historical numeric series are keyed by header name, not physical column.
            periods=list(forecast[(metric,next(k for m,k in matched if m==metric and k[0]==level))][0])
            for period in periods+[target]:
                aggregate=sum((forecast[(metric,tuple(row[c] for c in META))][0].get(period)
                              if period!=target else row['Actual']) or 0 for row in subset)
                time_series.append({'Metric':metric,'Market':level,'Period':period,'Actual':aggregate})
            last=[(r['Actual'],r['Last period']) for r in subset if r['Last period'] is not None]
            baselines.append({'Metric':metric,'Market':level,'Method':'Last period',**_score(last)})
            seasonal=[(r['Actual'],r['Seasonal previous year']) for r in subset
                      if r['Seasonal previous year'] is not None]
            baselines.append({'Metric':metric,'Market':level,'Method':'Seasonal naive',**_score(seasonal)})
            for dimension in GROUPS:
                groups=defaultdict(list)
                for row in subset: groups[row[dimension]].append(row)
                actual_total=sum(r['Actual'] for r in subset)
                forecast_total=sum(r['Forecast'] for r in subset)
                mat_periods=(periods+[target])[-6 if '-' in target else -12:]
                mat_totals={}
                for group, group_rows in groups.items():
                    score=_score([(r['Actual'],r['Forecast']) for r in group_rows])
                    a_share=100*score['actual']/actual_total if actual_total else None
                    f_share=100*score['forecast']/forecast_total if forecast_total else None
                    shares.append({'Metric':metric,'Market':level,'Dimension':dimension,
                                   'Group':group,'actual_share_pct':a_share,
                                   'forecast_share_pct':f_share,
                                   'share_delta_pp':f_share-a_share if a_share is not None and f_share is not None else None,
                                   **score})
                    mat_totals[group]=sum(sum((forecast[(metric,tuple(r[c] for c in META))][0].get(period)
                                                if period!=target else r['Actual']) or 0
                                                for period in mat_periods) for r in group_rows)
                mat_denominator=sum(mat_totals.values())
                for group, val in mat_totals.items():
                    mat_shares.append({'Metric':metric,'Market':level,'Dimension':dimension,
                                       'Group':group,'MAT actual':val,
                                       'MAT share %':100*val/mat_denominator if mat_denominator else None})
    _write_csv(output/'row_errors.csv',details,list(details[0]))
    _write_csv(output/'scorecard.csv',scores,list(scores[0]))
    _write_csv(output/'baseline.csv',baselines,list(baselines[0]))
    _write_csv(output/'market_shares.csv',shares,list(shares[0]))
    _write_csv(output/'mat_shares.csv',mat_shares,list(mat_shares[0]))
    _write_csv(output/'demand_segments.csv',demand_scores,list(demand_scores[0]))
    _write_csv(output/'time_series.csv',time_series,list(time_series[0]))
    (output/'reconciliation.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    return qa


def main(argv=None):
    parser=argparse.ArgumentParser(description='Compare Excel forecast with appended actual master sheet')
    parser.add_argument('workbook',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/forecast_validation'))
    args=parser.parse_args(argv)
    print(json.dumps(analyze(args.workbook,args.output),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
