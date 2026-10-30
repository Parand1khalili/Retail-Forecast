"""Behavioral checks: no future-label leakage, future-only dates and sparse data."""
import json,hashlib
import numpy as np
import pandas as pd
from app_core import Dataset,ROOT
from auto_engine import OUT,SELECTION_VERSION,candidates,fit_models,future_request,guarded_money,price_shift_guard

def main():
    d=Dataset.builtin();cutoff=d.months.index('2026-05')
    changed=Dataset(d.meta,d.volume.copy(),d.value.copy(),d.months,d.source)
    changed.volume[:,cutoff+1:]=1e12;changed.value[:,cutoff+1:]=5e15
    for metric in ['volume_kg','value_1000_toman']:
        a=candidates(d,metric,cutoff);b=candidates(changed,metric,cutoff)
        assert set(a)==set(b)
        assert all(np.array_equal(a[key],b[key]) for key in a),'Future actuals changed a prediction'
    assert price_shift_guard(d,cutoff)==price_shift_guard(changed,cutoff)
    assert price_shift_guard(d,cutoff)['active']
    for metric in ['volume_kg','value_1000_toman']:
        for h in [1,2,3]:
            r=future_request(d,{'metric':metric,'horizon':h,'filters':{'City':'Tehran','Brand':'Kalleh','SubBrand':'Seven','Subcategory':'Stirred'}})
            assert pd.Period(r['target'])>pd.Period(d.months[-1])
            assert r['actual_if_known'] is None and np.isfinite(r['prediction']) and r['prediction']>=0
    try:future_request(d,{'horizon':4})
    except ValueError:pass
    else:raise AssertionError('Unsupported horizons must fail explicitly')
    meta=np.array([['Tehran','Maker','Test','Plain','Fresh','LF','Not specified','Not specified'],
                   ['Karaj','Maker','Test','Plain','Fresh','LF','Not specified','Not specified']])
    dates=[str(m) for m in pd.period_range('2024-01',periods=18,freq='M')]
    v=np.repeat(np.array([[100.],[50.]]),18,axis=1)
    constant=Dataset(meta,v,v*10,dates,'synthetic constant series')
    p=fit_models(constant,'volume_kg',17,'all')
    assert np.allclose(p['raw'],v[:,-1,None])
    zero=Dataset(meta,np.zeros_like(v),np.zeros_like(v),dates,'synthetic unobserved series')
    assert fit_models(zero,'volume_kg',17,'all')=={}
    issued=OUT/'issued'/f'2026-08_{SELECTION_VERSION}.npz'
    audit=json.loads(issued.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(issued.read_bytes()).hexdigest()==audit['sha256']
    with np.load(issued) as z:
        assert not set(z['months'])&set(d.months)
        assert z['volume_kg'].shape==z['value_1000_toman'].shape==(len(d.meta),3)
        assert np.isfinite(z['volume_kg']).all() and np.isfinite(z['value_1000_toman']).all()
    print(json.dumps({'future_label_mutation':'passed','price_shift_guard':'passed','six_future_requests':'passed',
                      'constant_series':'passed','all_zero_series':'passed','invalid_horizon':'passed','issued_forecast_integrity':'passed'},ensure_ascii=True))

if __name__=='__main__':main()
