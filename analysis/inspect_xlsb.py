import sys, json
from pathlib import Path
from itertools import islice
sys.path.insert(0,str(Path(__file__).parent/'deps'))
from pyxlsb import open_workbook
root=Path("D:/Aug'26/Aug'26")
result=[]
for p in sorted(root.glob('*.xlsb')):
    entry={'file':p.name,'sheets':[]}
    with open_workbook(str(p)) as w:
        for name in w.sheets:
            with w.get_sheet(name) as s:
                rows=[]
                for row in islice(s.rows(),10):
                    rows.append({str(c.c+1):c.v for c in row if c.v is not None})
                entry['sheets'].append({'name':name,'dimension':str(s.dimension),'rows':rows})
    result.append(entry)
    print(json.dumps(entry,ensure_ascii=False),flush=True)
(Path(__file__).parent/'xlsb_structure.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
