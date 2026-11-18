import argparse
import asyncio
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp import web
from load_test import percentile, make_prompt, summarize, diagnostics, request_one

class TestMetrics(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile([1,2,3,4,5],95),5)
        self.assertEqual(percentile([1,2,3,4,5],50),3)
    def test_empty(self):
        self.assertIsNone(percentile([],95))
    def test_deterministic(self):
        self.assertEqual(make_prompt(1,2),make_prompt(1,2))
        self.assertNotEqual(make_prompt(1,2),make_prompt(1,3))
    def test_summary(self):
        rows=[{'status':'ok','output_tokens':20,'latency_s':2.,'ttft_s':.5,'decode_s':1.,'http_status':200},
              {'status':'error','latency_s':3.,'ttft_s':None,'output_tokens':None}]
        x=summarize(2,rows,10,14,[{'gpu_used_mib':1200,'gpu_util_pct':50}])
        self.assertEqual(x['completed'],1); self.assertEqual(x['output_tokens_per_s'],5)
        self.assertEqual(x['failure_rate'],.5)
    def test_diagnostics(self):
        v=[{'concurrency':1,'latency_p95_s':1,'output_tokens_per_s':10},
           {'concurrency':2,'latency_p95_s':3,'output_tokens_per_s':13,'failure_rate':0}]
        self.assertTrue(any('p95 latency' in s for s in diagnostics(v)))

class TestStreaming(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def generate(request):
            _=await request.json()
            response=web.StreamResponse(status=200,headers={'Content-Type':'application/x-ndjson'})
            await response.prepare(request)
            await response.write((json.dumps({'response':'Hello','done':False})+'\n').encode())
            await response.write((json.dumps({'response':' world','done':False})+'\n').encode())
            await response.write((json.dumps({'done':True,'eval_count':4,'eval_duration':1_000_000_000,'prompt_eval_count':12,'prompt_eval_duration':250_000_000,'total_duration':1_500_000_000})+'\n').encode())
            await response.write_eof();return response
        self.app=web.Application();self.app.router.add_post('/api/generate',generate)
        self.runner=web.AppRunner(self.app);await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.port=site._server.sockets[0].getsockname()[1]
    async def asyncTearDown(self):
        await self.runner.cleanup()
    async def test_stream(self):
        import aiohttp,time
        args=argparse.Namespace(base_url=f'http://127.0.0.1:{self.port}', model='mock',
                                keep_alive='20m',context=2048,max_tokens=20,seed=1,prompt_words=30)
        async with aiohttp.ClientSession() as session:
            result=await request_one(session,args,0,0,1,time.perf_counter(),[],asyncio.Lock())
        self.assertEqual(result['status'],'ok');self.assertEqual(result['text_chars'],11)
        self.assertEqual(result['output_tokens'],4);self.assertEqual(result['decode_s'],1)
        self.assertIsNotNone(result['ttft_s'])

if __name__=='__main__':unittest.main()
