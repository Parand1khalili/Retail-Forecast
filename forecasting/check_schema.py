"""Parser checks with tiny, explicitly synthetic workbooks."""
import json
import numpy as np
from app_core import Dataset,ROOT
from schema import SchemaError,metric_kind,month

def main():
    folder=ROOT/'outputs/schema_tests'
    d=Dataset.file(folder/'similar_headers.xlsx')
    assert d.months==['2026-05','2026-06','2026-07']
    assert len(d.meta)==2 and d.volume_unit=='لیتر' and d.value_unit=='تومان'
    assert d.volume[:,-1].sum()==180 and d.value[:,-1].sum()==2160
    assert set(d.meta[:,6])=={'Not specified'}
    try:Dataset.file(folder/'manual_headers.xlsx')
    except SchemaError as e:
        assert e.prompt['mapping_required'] and 'Place X' in e.prompt['headers']
    else:raise AssertionError('Unknown headers must request mapping')
    mapping={'fields':{'City':'Place X','Brand':'Label X','Company':'Owner X','SubBrand':'Family X',
                       'Subcategory':'Product X','Subcategory2':'Grade X','Measure':'Axis'},
             'measure_values':{'volume_kg':'Units moved','value_1000_toman':'Gross turnover'},
             'units':{'volume_kg':'لیتر','value_1000_toman':'تومان'}}
    m=Dataset.file(folder/'manual_headers.xlsx',mapping)
    assert np.array_equal(m.meta,d.meta) and np.array_equal(m.volume,d.volume) and np.array_equal(m.value,d.value)
    assert metric_kind('Volume Share') is None
    assert month('Aug\' 2026')=='2026-08' and month('2026/08')=='2026-08'
    print(json.dumps({'similar_headers':'passed','reordered_columns':'passed','header_row_3':'passed',
                      'manual_mapping':'passed','source_units':'passed','unknown_headers':'review required'},ensure_ascii=True))

if __name__=='__main__':main()
