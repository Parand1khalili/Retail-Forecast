"""Prepare future inference caches; these dates have no target actuals in file."""
import json,sys
from app_core import Dataset
from auto_engine import future_request,OUT

metric=sys.argv[1] if len(sys.argv)>1 else 'volume_kg'
d=Dataset.builtin()
results=[]
for h in [1,2,3]:
    result=future_request(d,{'metric':metric,'horizon':h,'filters':{'City':'Tehran','Company':'Kalleh','Brand':'Kalleh','SubBrand':'Seven','Subcategory':'Stirred'}})
    results.append(result)
(OUT/(metric+'_future_example.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'metric':metric,'targets':[r['target'] for r in results],'predictions':[r['prediction'] for r in results]},ensure_ascii=True),flush=True)
