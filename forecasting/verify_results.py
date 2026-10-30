from pathlib import Path
import json,math,zipfile
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/yoghurt_backtest'
r=json.loads((OUT/'report_data.json').read_text(encoding='utf-8'))
chosen=r['selection']['selected_model']
metrics={(v['level'],v['horizon']):v for v in r['holdout_metrics'] if v['model']==chosen}
p=OUT/'Yoghurt_Jun_Jul_Aug_2026_backtest.xlsx'
w=load_workbook(p,data_only=True,read_only=True)
summary=w['Summary']
assert summary['B5'].value==chosen
for h in [1,2,3]:
    rr=8+h;m=metrics[('Total',h)]
    assert math.isclose(summary.cell(rr,2).value,m['actual'],rel_tol=1e-9)
    assert math.isclose(summary.cell(rr,3).value,m['predicted'],rel_tol=1e-9)
    assert math.isclose(summary.cell(rr,6).value,metrics[('Segment',h)]['wape'],rel_tol=1e-8)
assert math.isclose(summary['F12'].value,metrics[('Segment',0)]['wape'],rel_tol=1e-9)
for sheet,level in [('Brands','Brand'),('Subbrands','SubBrand'),('Segments','Segment')]:
    ws=w[sheet];rows=ws.iter_rows(min_row=6,values_only=True);headers=next(rows);ix={h:i for i,h in enumerate(headers)}
    aa=pp=ae=0.;count=0
    for row in rows:
        a=row[ix['Actual kg']];p0=row[ix['Forecast kg']]
        error=row[ix['Error kg']];abserror=row[ix['Absolute error kg']]
        assert math.isclose(error,p0-a,abs_tol=1e-6)
        assert math.isclose(abserror,abs(p0-a),abs_tol=1e-6)
        if a==0:assert row[ix['Absolute error %']]=='n.a.'
        aa+=a;pp+=p0;ae+=abserror;count+=1
        assert all(not(isinstance(c,str) and c.startswith(('#REF!','#DIV/0!','#VALUE!','#NAME?','#NUM!','#SPILL!','#CALC!'))) for c in row)
    mm=metrics[(level,0)]
    assert math.isclose(aa,mm['actual'],rel_tol=1e-9)
    assert math.isclose(pp,mm['predicted'],rel_tol=1e-9)
    assert math.isclose(ae/aa,mm['wape'],rel_tol=1e-9)
    print(sheet,'rows',count,'WAPE',round(ae/aa*100,4))
with zipfile.ZipFile(p) as z:
    charts=[n for n in z.namelist() if '/charts/chart' in n and n.endswith('.xml')]
    assert len(charts)==1
    chart=z.read(charts[0]).decode()
    assert '346AA0' in chart and 'CC7A30' in chart
print('Verified cached formula values, independent error calculations, level totals, zero handling and chart colors.')
