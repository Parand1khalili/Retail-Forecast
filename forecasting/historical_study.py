"""Rolling-origin audit of simple forecasts, with history length as a controlled input.

Selection origins end in November 2025, so even 3-month labels end by February
2026. Later scored targets begin in March. Inputs never include later columns.
"""
from pathlib import Path
import csv, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'outputs/yoghurt_backtest/prepared_panel.npz'
OUT = ROOT / 'outputs/yoghurt_history_study'
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ['City', 'Company', 'Brand', 'SubBrand', 'Subcategory', 'Subcategory2', 'Subcategory3', 'Subcategory4']
WINDOWS = [2, 3, 6, 12, 24, 36, 'all']
METHODS = ['Last month', 'Mean 2', 'Mean 3', 'Mean 6', 'Exponential smoothing', 'Damped trend', 'Seasonal naive']

def forecast(history, method, horizon=1):
    n = history.shape[1]
    if method == 'Last month': return history[:, -1]
    if method.startswith('Mean '):
        k = int(method.split()[-1])
        return history[:, -k:].mean(axis=1) if n >= k else None
    if method == 'Exponential smoothing':
        level = history[:, 0].copy()
        for col in range(1, n): level = .5 * history[:, col] + .5 * level
        return level
    if method == 'Damped trend':
        if n < 6: return None
        recent = history[:, -3:].mean(axis=1)
        previous = history[:, -6:-3].mean(axis=1)
        slope = (recent-previous)/3
        last = history[:, -1]
        # Bound extrapolation for sparse brands and month-to-month panel noise.
        delta = np.clip(slope*horizon, -.15*last*horizon, .15*last*horizon)
        return np.maximum(last+delta, 0)
    raise ValueError(method)

def main():
    with np.load(PANEL) as z:
        meta = pd.DataFrame(z['meta'], columns=DIMS)
        dates = pd.PeriodIndex(z['months'], freq='M')
        arrays = {'volume_kg': z['volume'], 'value_1000_toman': z['value']}
    group_columns = {
        'total': [],
        'brand': ['Company', 'Brand'],
        'city_subbrand': ['City', 'Company', 'Brand', 'SubBrand'],
        'city_segment': DIMS,
    }
    grouped = {}
    for level, cols in group_columns.items():
        ids = meta.groupby(cols, sort=False).ngroup().to_numpy() if cols else np.zeros(len(meta), dtype=int)
        grouped[level] = {}
        for metric, array in arrays.items():
            out = np.zeros((ids.max()+1, array.shape[1]), dtype=float)
            np.add.at(out, ids, array)
            grouped[level][metric] = out
    records = []
    first_origin = dates.get_loc(pd.Period('2024-03'))
    last_origin = dates.get_loc(pd.Period('2026-05'))
    for metric in arrays:
        for level in group_columns:
            y = grouped[level][metric]
            for origin in range(first_origin, last_origin+1):
                origin_date = str(dates[origin])
                phase = 'selection' if dates[origin] <= pd.Period('2025-11') else ('later' if dates[origin]>=pd.Period('2026-02') else 'bridge')
                for window in WINDOWS:
                    start = max(0, origin+1-window) if isinstance(window, int) else 0
                    hist = y[:, start:origin+1]
                    if isinstance(window, int) and hist.shape[1] < window: continue
                    for method in METHODS:
                        if method == 'Seasonal naive':
                            if hist.shape[1] < 12: continue
                            for h in [1, 2, 3]:
                                actual = y[:, origin+h]
                                pred = hist[:, -12+h-1]
                                records.append((metric, level, phase, origin_date, str(dates[origin+h]), h, str(window), method, len(hist.T), float(np.abs(pred-actual).sum()), float(actual.sum()), float((pred-actual).sum())))
                            continue
                        for h in [1, 2, 3]:
                            pred = forecast(hist, method, h)
                            if pred is None: continue
                            actual = y[:, origin+h]
                            records.append((metric, level, phase, origin_date, str(dates[origin+h]), h, str(window), method, len(hist.T), float(np.abs(pred-actual).sum()), float(actual.sum()), float((pred-actual).sum())))
            print('DONE', metric, level, flush=True)
    columns = ['metric','level','phase','origin','target','horizon','input_months','method','actual_input_months','absolute_error','actual','signed_error']
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(OUT/'rolling_detail.csv', index=False, encoding='utf-8-sig')
    common = df[(df.phase=='selection') & (df.origin>='2025-02') & (df.origin<='2025-11')]
    def summary(data):
        s = data.groupby(['metric','level','horizon','input_months','method'],as_index=False).agg(origins=('origin','nunique'),absolute_error=('absolute_error','sum'),actual=('actual','sum'),signed_error=('signed_error','sum'))
        s['wape'] = s.absolute_error/s.actual
        s['bias'] = s.signed_error/s.actual
        return s
    selection = summary(common)
    selection.to_csv(OUT/'window_comparison.csv', index=False, encoding='utf-8-sig')
    selected = {}
    # Choose a single history length and method for each metric/horizon at the
    # exact city+taxonomy grain the UI can request. No later outcomes enter here.
    for metric in arrays:
        selected[metric] = {}
        for h in [1,2,3]:
            s = selection[(selection.metric==metric)&(selection.level=='city_segment')&(selection.horizon==h)&(selection.origins==10)].copy()
            s['required_months'] = s.input_months.map(lambda x: 54 if x=='all' else int(x))
            s = s.sort_values(['wape','required_months','method'])
            row = s.iloc[0]
            selected[metric][str(h)] = {'method':row['method'],'input_months':row['input_months'],'selection_wape':float(row.wape)}
    (OUT/'selection_locked.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    # Review the preselected recipe on later origins and by several useful levels.
    chosen_rows=[]
    for metric in arrays:
        for h in [1,2,3]:
            picked=selected[metric][str(h)]
            sel=df[(df.metric==metric)&(df.horizon==h)&(df.method==picked['method'])&(df.input_months==picked['input_months'])&(df.phase=='later')]
            for level in group_columns:
                z=sel[sel.level==level]
                for origin,part in z.groupby('origin'):
                    chosen_rows.append({'metric':metric,'level':level,'horizon':h,'origin':origin,'target':part.iloc[0].target,'method':picked['method'],'input_months':picked['input_months'],'wape':float(part.absolute_error.sum()/part.actual.sum()),'bias':float(part.signed_error.sum()/part.actual.sum()),'actual':float(part.actual.sum())})
    chosen=pd.DataFrame(chosen_rows)
    chosen.to_csv(OUT/'locked_later_results.csv',index=False,encoding='utf-8-sig')
    brief={'dates':[str(dates[0]),str(dates[-1])], 'selection_origins':['2025-02','2025-11'], 'selection_target_end':'2026-02', 'later_origins':['2026-02','2026-05'], 'later_target_start':'2026-03', 'selection':selected,
           'later_city_segment':chosen[chosen.level=='city_segment'].to_dict('records'),
           'input_policy':'input_months counts monthly report columns. Their underlying panel audit window overlaps. All months are published estimates.',
           'scoring':'WAPE = sum of absolute city-segment errors / sum of actual sales. Zero actual rows remain in numerator.'}
    (OUT/'study.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selection':selected,'later_rows':len(brief['later_city_segment'])},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
