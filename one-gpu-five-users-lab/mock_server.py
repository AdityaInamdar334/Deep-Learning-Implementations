#!/usr/bin/env python3
"""Mock local Ollama API for testing plumbing. NOT model performance data."""
import asyncio
import json
from aiohttp import web

async def generate(request):
    data=await request.json()
    if not data.get('stream',True):
        return web.json_response({'response':'OK','done':True,'eval_count':1,'eval_duration':100000000})
    response=web.StreamResponse(headers={'Content-Type':'application/x-ndjson'})
    await response.prepare(request)
    await asyncio.sleep(.015)
    await response.write((json.dumps({'response':'Synthetic text ','done':False})+'\n').encode())
    await asyncio.sleep(.02)
    await response.write((json.dumps({'response':'only','done':False})+'\n').encode())
    await response.write((json.dumps({'done':True,'eval_count':4,'eval_duration':20_000_000,
                                      'prompt_eval_count':12,'prompt_eval_duration':15_000_000,
                                      'load_duration':0,'total_duration':35_000_000})+'\n').encode())
    await response.write_eof();return response

async def version(request):return web.json_response({'version':'mock-no-llm'})
async def ps(request):return web.json_response({'models':[{'name':'mock','size_vram':0,'context_length':2048}]})

app=web.Application(); app.router.add_post('/api/generate',generate)
app.router.add_get('/api/version',version);app.router.add_get('/api/ps',ps)
if __name__=='__main__':web.run_app(app,host='127.0.0.1',port=11435)
