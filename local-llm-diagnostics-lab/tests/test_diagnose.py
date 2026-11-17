import json,sys,unittest,threading,urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch, MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose import stream_generate, diagnose, safe_ratio

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length=int(self.headers['Content-Length']); payload=json.loads(self.rfile.read(length))
        self.send_response(200); self.send_header('Content-Type','application/x-ndjson'); self.end_headers()
        for chunk in [{'response':'Hello','done':False},{'response':' world','done':False},
                      {'response':'','done':True,'load_duration':10000000,'total_duration':500000000,
                       'prompt_eval_count':20,'prompt_eval_cached_count':4,'prompt_eval_duration':100000000,
                       'eval_count':2,'eval_duration':200000000}]:
            self.wfile.write((json.dumps(chunk)+'\n').encode()); self.wfile.flush()
    def log_message(self,*args): pass

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        except OSError:
            cls.server=None
    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.shutdown();cls.server.server_close()
    def test_stream_ttft_and_throughput(self):
        if self.server:
            try:
                r=stream_generate('http://127.0.0.1:'+str(self.server.server_port),'mock','hi',2048,4,10,42)
            except (urllib.error.URLError, OSError):
                r=None
        else:
            r=None
        if r is None:
            mock_data = [
                json.dumps({'response':'Hello','done':False}).encode() + b'\n',
                json.dumps({'response':' world','done':False}).encode() + b'\n',
                json.dumps({'response':'','done':True,'load_duration':10000000,'total_duration':500000000,
                            'prompt_eval_count':20,'prompt_eval_cached_count':4,'prompt_eval_duration':100000000,
                            'eval_count':2,'eval_duration':200000000}).encode() + b'\n'
            ]
            with patch('urllib.request.urlopen') as mock_open:
                mock_cm = MagicMock()
                mock_cm.__enter__.return_value = mock_data
                mock_open.return_value = mock_cm
                r=stream_generate('http://127.0.0.1:0','mock','hi',2048,4,10,42)
        self.assertEqual(r['output'],'Hello world');self.assertEqual(r['prefill_tps'],160)
        self.assertEqual(r['decode_tps'],10);self.assertIsNotNone(r['client_ttft_s']);self.assertEqual(r['cached_prompt_tokens'],4)
    def test_ratio(self): self.assertIsNone(safe_ratio(1,0))
    def test_load(self):
        f,_=diagnose({'load_s':2,'wall_s':3,'generated_tokens':2},None)
        self.assertIn('LOAD_DOMINATED',f)
    def test_offload(self):
        f,_=diagnose({'wall_s':1}, {'size_bytes':100,'size_vram_bytes':40})
        self.assertIn('POSSIBLE_CPU_OFFLOAD',f)
    def test_vram(self):
        f,_=diagnose({'wall_s':1,'gpu_samples':[{'used_mib':7400,'total_mib':8000,'gpu_util_pct':40}]},None)
        self.assertIn('HIGH_VRAM_PRESSURE',f)
    def test_missing_ttft(self):
        f,_=diagnose({'wall_s':1,'client_ttft_s':None},None)
        self.assertIn('NO_VISIBLE_TOKEN',f)
    def test_prefill(self):
        f,_=diagnose({'server_prefill_s':5,'server_decode_s':1,'prompt_tokens':1000},None)
        self.assertIn('PREFILL_DOMINATED',f)
    def test_decode(self):
        f,_=diagnose({'server_prefill_s':1,'server_decode_s':6,'generated_tokens':100},None)
        self.assertIn('DECODE_DOMINATED',f)

if __name__=='__main__':unittest.main()
