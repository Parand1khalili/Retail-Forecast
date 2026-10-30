from pathlib import Path
from itertools import islice
import sys,json
sys.path.insert(0,str(Path(__file__).parent/'deps'))
from pyxlsb import open_workbook
p=Path("D:/Aug'26/Aug'26/Yoghurt Kalleh Aug' 2026 City Part 1.xlsb")
with open_workbook(str(p)) as w:
    with w.get_sheet('City1') as s:
        result=[]
        for row in islice(s.rows(),110):
            d={c.c:c.v for c in row if c.v is not None}
            if 13 in d and d[13]!='Index Name':
                result.append({'row':row[0].r+1,'level':d.get(2),'description':d.get(3),'metric':d[13],'latest':d.get(57)})
out=Path(__file__).parent/'raw_metrics_sample.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result[:50],ensure_ascii=False))
