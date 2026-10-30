from pathlib import Path
import zipfile,xml.etree.ElementTree as E,json,re
p=next(Path("D:/Aug'26/Aug'26").glob('*.pptx'))
with zipfile.ZipFile(p) as z:
    names=sorted([n for n in z.namelist() if re.match(r'ppt/slides/slide[0-9]+[.]xml$',n)],key=lambda n:int(re.search(r'slide([0-9]+)',n).group(1)))
    out=[{'slide':i+1,'text':'\n'.join(e.text or '' for e in E.fromstring(z.read(n)).iter() if e.tag.endswith('}t'))} for i,n in enumerate(names)]
    notes=[]
    for n in z.namelist():
        if re.match(r'ppt/notesSlides/notesSlide[0-9]+[.]xml$',n):
            txt='\n'.join(e.text or '' for e in E.fromstring(z.read(n)).iter() if e.tag.endswith('}t'))
            if len(txt)>30:notes.append({'part':n,'text':txt})
Path('analysis/deck_text.json').write_text(json.dumps({'slides':out,'notes':notes},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out[:15],ensure_ascii=False,indent=2))
print('METHODOLOGY MATCHES')
for s in out:
    if re.search(r'panel|sample|smooth|method|bimonth|bi.month|rolling|moving|audit|universe|coverage|نمونه|پنل|دوماه|ماهانه',s['text'],re.I):print(json.dumps(s,ensure_ascii=False))
