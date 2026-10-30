"""Integration checks against the supplied master and the frozen audit panel."""
from pathlib import Path
import json, sys
import numpy as np
from app_core import Dataset, predict

SOURCE=Path("D:/Aug'26/Aug'26/Master Mast- Aug' 25.xlsx")

def main():
    original=Dataset.file(SOURCE)
    reference=Dataset.builtin()
    assert original.months==reference.months
    oi=np.lexsort(original.meta.T[::-1]);ri=np.lexsort(reference.meta.T[::-1])
    assert np.array_equal(original.meta[oi],reference.meta[ri])
    vdiff=float(np.max(np.abs(original.volume[oi]-reference.volume[ri])))
    moneydiff=float(np.max(np.abs(original.value[oi]-reference.value[ri])))
    assert vdiff<1e-5 and moneydiff<1e-4,(vdiff,moneydiff)
    q={'metric':'volume_kg','cutoff':'2026-05','horizon':3,'window':2,'filters':{'City':'Tehran','Subcategory':'Stirred'}}
    old=predict(original,q)
    altered=Dataset(original.meta,original.volume.copy(),original.value.copy(),original.months,original.source)
    altered.volume[:,original.months.index('2026-06'):]=1e12
    changed=predict(altered,q)
    assert old['prediction']==changed['prediction'], 'Future volume leaked into May-origin forecast'
    assert old['method']==changed['method'], 'Future volume leaked into model choice'
    qmoney=dict(q,metric='value_1000_toman',window=6)
    past_money=predict(original,qmoney)
    altered.value[:,original.months.index('2026-06'):]=1e12
    changed_money=predict(altered,qmoney)
    assert past_money['prediction']==changed_money['prediction'], 'Future value leaked into May-origin forecast'
    print(json.dumps({'source_rows':len(original.meta),'months':len(original.months),
                      'max_volume_difference':vdiff,'max_value_difference':moneydiff,
                      'tehran_stirred_aug_forecast_kg':old['prediction'],
                      'tehran_stirred_aug_actual_kg':old['actual_if_known'],
                      'tehran_stirred_aug_forecast_1000_toman':past_money['prediction'],
                      'future_leakage_guard':'passed'},ensure_ascii=True))

if __name__=='__main__':main()
