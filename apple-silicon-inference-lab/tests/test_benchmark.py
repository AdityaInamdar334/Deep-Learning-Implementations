import unittest,sys,pathlib,json,io
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark import request_config,post_stream,proc_rss_mib
from unittest.mock import patch
class FakeResponse:
 def __init__(self,lines):self.lines=lines
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def __iter__(self):return iter(self.lines)
class Tests(unittest.TestCase):
 def test_ollama_payload(self):
  u,p=request_config('ollama','x','hello',100,5,4096);self.assertTrue(p['raw']);self.assertEqual(p['options']['num_ctx'],4096)
 def test_mlx_url(self):self.assertIn(':8081/',request_config('mlx','x','hi',50,1,512)[0])
 def test_llama_url(self):self.assertIn(':8080/',request_config('llama','x','hi',50,1,512)[0])
 @patch('urllib.request.urlopen')
 def test_ollama_stream(self,fake):
  events=[{'response':'Hello','done':False},{'response':' world','done':False},{'response':'','done':True,'eval_count':2,'eval_duration':1000000000,'prompt_eval_count':10,'prompt_eval_duration':1000000000}]
  fake.return_value=FakeResponse([(json.dumps(x)+'\n').encode() for x in events]);o=post_stream('http://test',{'stream':True},'ollama');self.assertEqual(o['text'],'Hello world');self.assertEqual(o['decode_tps'],2);self.assertEqual(o['prefill_tps'],10);self.assertIsNotNone(o['ttft_visible_s'])
 @patch('urllib.request.urlopen')
 def test_openai_sse(self,fake):
  lines=[b'data: {"choices":[{"text":"Hi"}]}\n',b'data: {"choices":[],"usage":{"completion_tokens":1}}\n',b'data: [DONE]\n'];fake.return_value=FakeResponse(lines);o=post_stream('http://test',{},'mlx');self.assertEqual(o['text'],'Hi');self.assertEqual(o['completion_tokens'],1);self.assertIsNone(o['decode_tps'])
 def test_rss_missing(self):self.assertIsNone(proc_rss_mib(None))
if __name__=='__main__':unittest.main()
