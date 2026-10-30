from pathlib import Path
import zipfile, xml.etree.ElementTree as ET, json
from itertools import islice

ROOT = Path("D:/Aug'26/Aug'26")
NS = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
OUT = Path(__file__).parent

def inspect_xlsx(p):
    with zipfile.ZipFile(p) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets={r.attrib['Id']:r.attrib['Target'] for r in rels}
        ss=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                for _, el in ET.iterparse(f, events=('end',)):
                    if el.tag.endswith('}si'):
                        ss.append(''.join(t.text or '' for t in el.iter() if t.tag.endswith('}t')))
                        el.clear()
        result={'file':p.name,'shared_strings':len(ss),'sheets':[]}
        for sh in wb.find('m:sheets',NS):
            rid=sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            t=targets[rid].replace('\\','/')
            name=t.lstrip('/') if t.startswith('/') else 'xl/'+t
            meta={'name':sh.attrib['name'],'state':sh.attrib.get('state'),'xml':name,'xml_size':z.getinfo(name).file_size,'rows':[]}
            with z.open(name) as f:
                for _, el in ET.iterparse(f,events=('end',)):
                    if el.tag.endswith('}dimension'): meta['dimension']=el.attrib
                    if el.tag.endswith('}row'):
                        row={}
                        for c in el:
                            v=c.find('m:v',NS)
                            val=v.text if v is not None else None
                            if c.attrib.get('t')=='s' and val is not None: val=ss[int(val)]
                            elif c.attrib.get('t')=='inlineStr': val=''.join(t.text or '' for t in c.iter() if t.tag.endswith('}t'))
                            if val is not None: row[c.attrib['r']]=val
                        meta['rows'].append(row)
                        el.clear()
                        if len(meta['rows'])>=9: break
            result['sheets'].append(meta)
        return result

for p in sorted(ROOT.glob('*.xlsx')):
    print('READING',p.name,flush=True)
    r=inspect_xlsx(p)
    (OUT/(p.stem+'_structure.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False),flush=True)

for p in sorted(ROOT.glob('*.xlsb')):
    with zipfile.ZipFile(p) as z:
        print(json.dumps({'file':p.name,'parts':[(i.filename,i.file_size) for i in z.infolist() if ('worksheets/sheet' in i.filename and i.filename.endswith('.bin')) or i.filename in ['xl/workbook.bin','xl/sharedStrings.bin']]}),flush=True)
