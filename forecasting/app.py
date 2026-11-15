"""Run `python forecasting/app.py` then open http://127.0.0.1:8765 ."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs
import json, threading, uuid, time, zipfile, os, sys
from runtime_preflight import diagnose, require_ready

DEPENDENCIES = diagnose()
if any(issue['package'] in ('numpy', 'pandas') for issue in DEPENDENCIES['failures']):
    # app_core needs these packages for even the import/upload UI.
    print('EMRC Forecast: mandatory Python dependencies are unavailable.', file=sys.stderr)
    print(DEPENDENCIES['install_command'].encode('ascii','backslashreplace').decode('ascii'),file=sys.stderr)
    raise SystemExit(2)

from app_core import Dataset, predict, ROOT
from schema import SchemaError
from runtime_identity import identity

RUNTIME = identity()
RUNTIME['pid'] = os.getpid()
RUNTIME['dependencies'] = DEPENDENCIES
RUNTIME['paths'].update({'app.py': str(Path(__file__).resolve()),
                        'app_core.py': str(__import__('app_core').__file__),
                        'schema.py': str(__import__('schema').__file__)})

HOST='127.0.0.1';PORT=8765
HERE=Path(__file__).parent
UPLOADS=ROOT/'outputs/forecast_app/uploads'
UPLOADS.mkdir(parents=True,exist_ok=True)
try:
    initial_dataset=Dataset.builtin()
except FileNotFoundError:
    initial_dataset=None
state={'dataset':initial_dataset,'benchmark_zip':None}
lock=threading.RLock()
training_lock=threading.Lock()

def cleanup_upload(path):
    """Best-effort cleanup; a transient Windows file lock must not fail an upload."""
    for attempt in range(6):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    # Windows antivirus / file indexers may release the XLSX later.
    def retry():
        for _ in range(30):
            time.sleep(1)
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError:
                pass
        print(f'WARNING: temporary upload still locked; delete when unlocked: {path}',flush=True)
    threading.Thread(target=retry,daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):print('%s %s'%(self.address_string(),fmt%args),flush=True)

    def answer(self,data,status=200):
        raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/health':
            self.answer({'service':'ra-forecast','ready':DEPENDENCIES['ready'],'runtime':RUNTIME});return
        if path=='/api/backtest/report':
            with lock: archive=state.get('benchmark_zip')
            if archive is None:
                self.answer({'error':'ابتدا ارزیابی مدل‌ها را برای فایل فعلی اجرا کنید.'},400);return
            self.send_response(200)
            self.send_header('Content-Type','application/zip')
            self.send_header('Content-Disposition','attachment; filename="emrc_model_backtest.zip"')
            self.send_header('Content-Length',str(len(archive)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            for start in range(0,len(archive),1024*1024):
                self.wfile.write(archive[start:start+1024*1024])
            return
        if path=='/api/export/full':
            if not DEPENDENCIES['ready']:
                self.answer({'error':DEPENDENCIES['message'],'dependency_check':DEPENDENCIES},503);return
            import tempfile
            strategy=parse_qs(urlparse(self.path).query).get('strategy',['hierarchical'])[0]
            if strategy not in ('hierarchical','global'):
                self.answer({'error':'روش انتخاب مدل نامعتبر است.'},400);return
            from full_export import export_workbook
            with lock: dataset=state['dataset']
            if dataset is None:
                self.answer({'error':'ابتدا فایل مستر را بارگذاری کنید.'},400);return
            try:
                # Build completely before sending headers; a failed model must
                # never be returned as a corrupt or partially written XLSX.
                with training_lock:
                    with tempfile.TemporaryDirectory(prefix='emrc_export_') as folder:
                        workbook=Path(folder)/'retail_forecast_full.xlsx'
                        export_workbook(dataset,workbook,strategy=strategy)
                        self.send_response(200)
                        self.send_header('Content-Type','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        self.send_header('Content-Disposition','attachment; filename="retail_forecast_full.xlsx"')
                        self.send_header('Content-Length',str(workbook.stat().st_size))
                        self.send_header('Cache-Control','no-store')
                        self.end_headers()
                        with workbook.open('rb') as stream:
                            for chunk in iter(lambda:stream.read(1024*1024),b''):
                                self.wfile.write(chunk)
            except Exception as e:
                self.answer({'error':str(e)},400)
            return
        if path=='/api/data':
            with lock:dataset=state['dataset']
            self.answer(dataset.options() if dataset is not None else {
                'ready':False,'message':'فایل نمونه در این رایانه موجود نیست؛ فایل مستر خود را بارگذاری کنید.'
            })
            return
        if path in ['/','/index.html']:
            raw=(HERE/'app_ui.html').read_bytes()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>220*1024*1024:raise ValueError('حجم فایل باید کمتر از ۲۲۰ مگابایت باشد.')
            path=urlparse(self.path).path
            if path=='/api/upload':
                name=unquote(self.headers.get('X-Filename','uploaded.xlsx'))
                if not name.lower().endswith('.xlsx'):raise ValueError('فقط فایل xlsx مستر پشتیبانی می‌شود.')
                raw_mapping=self.headers.get('X-Mapping')
                mapping=json.loads(unquote(raw_mapping)) if raw_mapping else {}
                mapping['periodicity']=self.headers.get('X-Periodicity','monthly')
                dest=UPLOADS/(uuid.uuid4().hex+'.xlsx')
                with dest.open('wb') as f:
                    left=n
                    while left:
                        chunk=self.rfile.read(min(left,4*1024*1024))
                        if not chunk:raise ValueError('انتقال فایل کامل نشد.')
                        f.write(chunk);left-=len(chunk)
                try:
                    dataset=Dataset.file(dest,mapping);dataset.source=name
                finally:
                    cleanup_upload(dest)
                with lock:
                    state['dataset']=dataset
                    state['benchmark_zip']=None
                self.answer(dataset.options());return
            body=json.loads(self.rfile.read(n))
            if path=='/api/load':
                if body.get('builtin'):
                    dataset=Dataset.builtin()
                else:
                    loc=Path(str(body.get('path','')).strip().strip('"').strip("'"))
                    if not loc.is_file() or loc.suffix.lower()!='.xlsx':raise ValueError('مسیر فایل xlsx معتبر نیست.')
                    mapping=dict(body.get('mapping') or {})
                    mapping['periodicity']=body.get('periodicity',mapping.get('periodicity','monthly'))
                    dataset=Dataset.file(loc,mapping)
                with lock:
                    state['dataset']=dataset
                    state['benchmark_zip']=None
                self.answer(dataset.options());return
            if path=='/api/backtest':
                if not DEPENDENCIES['ready']:
                    self.answer({'error':DEPENDENCIES['message'],'dependency_check':DEPENDENCIES},503);return
                from backtest_lab import run_backtest
                with lock: dataset=state['dataset']
                if dataset is None:
                    raise ValueError('ابتدا فایل Excel را وارد و ستون‌های آن را تأیید کنید.')
                chosen=dataset.for_market(body.get('market')).for_grain(body.get('grain'))
                with training_lock:
                    report_data, archive=run_backtest(chosen,body.get('metrics'))
                with lock:
                    if state['dataset'] is not dataset:
                        self.answer({'error':'فایل ورودی در حین آزمون تغییر کرده است؛ دوباره آزمون را اجرا کنید.'},409)
                        return
                    state['benchmark_zip']=archive
                self.answer(report_data);return
            if path=='/api/forecast':
                with lock:dataset=state['dataset']
                self.answer(predict(dataset,body));return
            if path=='/api/predict':
                if not DEPENDENCIES['ready']:
                    self.answer({'error':DEPENDENCIES['message'],'dependency_check':DEPENDENCIES},503);return
                from auto_engine import future_request
                with lock:dataset=state['dataset']
                if dataset is None:raise ValueError('ابتدا یک فایل مستر بارگذاری کنید.')
                with training_lock:result=future_request(dataset,body)
                self.answer(result);return
            self.send_error(404)
        except SchemaError as e:
            self.answer(e.prompt,422)
        except zipfile.BadZipFile:
            self.answer({'error':'فایل Excel باز نشد؛ فایل باید یک مستر سالم با قالب xlsx باشد.'},400)
        except Exception as e:
            self.answer({'error':str(e)},400)

if __name__=='__main__':
    if not DEPENDENCIES['ready']:
        print("WARNING: EMRC Forecast CatBoost engine is not available in this Python.",flush=True)
        print(('Python: '+DEPENDENCIES['python_executable']).encode('ascii','backslashreplace').decode('ascii'),flush=True)
        print(('Install: '+DEPENDENCIES['install_command']).encode('ascii','backslashreplace').decode('ascii'),flush=True)
    try:
        server=ThreadingHTTPServer((HOST,PORT),Handler)
    except OSError as exc:
        raise SystemExit(
            f'EMRC Forecast cannot start: port {PORT} is already in use '
            f'or unavailable ({exc}). Stop the verified previous app.py '
            'instance, then run python app.py again. '
            'To inspect the port owner in PowerShell: '
            f'Get-NetTCPConnection -LocalPort {PORT} -State Listen'
        ) from exc
    print(f'RA forecast: http://{HOST}:{PORT}',flush=True)
    print(f'Active Python: {DEPENDENCIES["python_executable"]}',flush=True)
    print(f'Project: {ROOT}',flush=True)
    server.serve_forever()
