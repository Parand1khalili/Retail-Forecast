from pathlib import Path
import zipfile,sys
base=Path(__file__).parent
for wheel in (base/'wheels').glob('*.whl'):
    with zipfile.ZipFile(wheel) as z:z.extractall(base/'model_deps')
sys.path.insert(0,str(base/'model_deps'))
from catboost import CatBoostRegressor
import catboost,sklearn,scipy
print('READY',catboost.__version__,sklearn.__version__,scipy.__version__)
