from pathlib import Path
from pypdf import PdfReader
import json,re,shutil,importlib.util
p=Path("D:/Aug'26/Aug'26/RA Training.pdf")
r=PdfReader(p)
pages=[{'page':i+1,'text':page.extract_text() or ''} for i,page in enumerate(r.pages)]
Path('analysis/ra_training_text.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
print('PAGE_COUNT',len(pages))
for pg in pages:
    print(json.dumps(pg,ensure_ascii=False))
print('RENDER_TOOLS',json.dumps({k:shutil.which(k) for k in ['pdftoppm','mutool']}))
print('MODULES',json.dumps({k:importlib.util.find_spec(k) is not None for k in ['fitz','pypdfium2','pdfplumber']}))
