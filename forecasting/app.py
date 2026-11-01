"""Run `python forecasting/app.py` then open http://127.0.0.1:8765 ."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote
import json, threading, uuid, time
from app_core import Dataset, predict, ROOT
from schema import SchemaError

HOST='127.0.0.1';PORT=8765
HERE=Path(__file__).parent
UPLOADS=ROOT/'outputs/forecast_app/uploads'
UPLOADS.mkdir(parents=True,exist_ok=True)
try:
    initial_dataset=Dataset.builtin()
except FileNotFoundError:
    initial_dataset=None
state={'dataset':initial_dataset}
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
            self.answer({'service':'ra-forecast','ready':True});return
        if path=='/api/data':
            with lock:dataset=state['dataset']
            self.answer(dataset.options() if dataset is not None else {
                'ready':False,'message':'فایل نمونه در این رایانه موجود نیست؛ فایل مستر خود را بارگذاری کنید.'
            })
            return
        if path in ['/','/index.html']:
            raw=(HERE/'app_ui.html').read_bytes()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
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
                mapping=json.loads(unquote(raw_mapping)) if raw_mapping else None
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
                with lock:state['dataset']=dataset
                self.answer(dataset.options());return
            body=json.loads(self.rfile.read(n))
            if path=='/api/load':
                if body.get('builtin'):
                    dataset=Dataset.builtin()
                else:
                    loc=Path(str(body.get('path','')).strip().strip('"').strip("'"))
                    if not loc.is_file() or loc.suffix.lower()!='.xlsx':raise ValueError('مسیر فایل xlsx معتبر نیست.')
                    dataset=Dataset.file(loc,body.get('mapping'))
                with lock:state['dataset']=dataset
                self.answer(dataset.options());return
            if path=='/api/forecast':
                with lock:dataset=state['dataset']
                self.answer(predict(dataset,body));return
            if path=='/api/predict':
                from auto_engine import future_request
                with lock:dataset=state['dataset']
                if dataset is None:raise ValueError('ابتدا یک فایل مستر بارگذاری کنید.')
                with training_lock:result=future_request(dataset,body)
                self.answer(result);return
            self.send_error(404)
        except SchemaError as e:
            self.answer(e.prompt,422)
        except Exception as e:
            self.answer({'error':str(e)},400)

if __name__=='__main__':
    server=ThreadingHTTPServer((HOST,PORT),Handler)
    print(f'RA forecast: http://{HOST}:{PORT}',flush=True)
    server.serve_forever()
