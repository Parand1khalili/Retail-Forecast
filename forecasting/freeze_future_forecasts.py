"""Lock forecasts before the corresponding RA report columns are available."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from app_core import Dataset
from auto_engine import OUT,SELECTION_VERSION,automatic_forecast

def main():
    d=Dataset.builtin();folder=OUT/'issued';folder.mkdir(exist_ok=True)
    stem=d.months[-1]+'_'+SELECTION_VERSION
    target=folder/(stem+'.npz');description=folder/(stem+'.json')
    if target.exists() and description.exists():
        print('Issued forecast already exists; preserved without overwrite.');return
    months=[str(pd.Period(d.months[-1],freq='M')+h) for h in [1,2,3]]
    arrays={};recipes={}
    for metric in ['volume_kg','value_1000_toman']:
        predictions=[];recipes[metric]={}
        for h in [1,2,3]:
            p,recipe=automatic_forecast(d,metric,h)
            predictions.append(p);recipes[metric][str(h)]=recipe
        arrays[metric]=np.stack(predictions,axis=1)
    np.savez_compressed(target,**arrays,meta=d.meta,months=np.array(months),source_end=d.months[-1])
    audit={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_end':d.months[-1],
           'targets':months,'selection_version':SELECTION_VERSION,'rows':len(d.meta),'recipes':recipes,
           'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
           'note':'Targets absent from source file. Earlier calendar months may be unpublished RA nowcasts because publication is delayed. Keep this file unchanged when actual reports arrive.'}
    description.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(target),'targets':months,'rows':len(d.meta),'sha256':audit['sha256']},ensure_ascii=True))

if __name__=='__main__':main()
