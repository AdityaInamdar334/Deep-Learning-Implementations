#!/usr/bin/env python3
"""Closed-loop concurrency benchmark for local Ollama /api/generate, no cloud APIs."""
import argparse
import asyncio
import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

LEVELS = [1, 2, 4, 8, 16]


def percentile(values, p):
    """Nearest-rank percentile (no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered)-1, math.ceil(p / 100 * len(ordered))-1))]


def avg(values):
    return statistics.mean(values) if values else None


def make_prompt(user_id, request_id, words=140):
    # Deterministic and per-request distinct; shared header may still be cached.
    words = max(20, words)
    nonce = f"U{user_id:02d}-R{request_id:04d}"
    body = ' '.join(f"entry{(i * 37 + user_id * 11 + request_id * 19) % 997}" for i in range(words))
    return ("You are evaluating a local inference server. Reply with a concise description "
            "of the synthetic diagnostics below, in plain English. Avoid markdown. "
            f"Request tag: {nonce}. Data: {body}. Summarize two patterns.")


class Monitor:
    def __init__(self, base_url, gpu_index, interval):
        self.base_url = base_url.rstrip('/')
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.stop = asyncio.Event()

    async def run(self, session, start):
        gpu_exe = shutil.which('nvidia-smi')
        while not self.stop.is_set():
            sample = {'t_s': round(time.perf_counter()-start, 3)}
            if gpu_exe:
                try:
                    cmd = [gpu_exe, '-i', str(self.gpu_index), '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
                    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=4)
                    pieces = stdout.decode().strip().splitlines()[0].split(',')
                    if len(pieces) >= 3:
                        sample.update(gpu_used_mib=float(pieces[0]), gpu_total_mib=float(pieces[1]), gpu_util_pct=float(pieces[2]))
                except (ValueError, IndexError, OSError, asyncio.TimeoutError):
                    pass
            try:
                async with session.get(self.base_url + '/api/ps', timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get('models', [])
                        if models:
                            sample['ollama_vram_mib'] = sum(m.get('size_vram', 0) for m in models)/1048576
                            sample['ollama_loaded_models'] = len(models)
                            sample['ollama_context_lengths'] = [m.get('context_length') for m in models]
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                pass
            self.samples.append(sample)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass


async def request_one(session, args, user_id, seq, level, start, event_trace, lock):
    started = time.perf_counter()
    result = dict(concurrency=level, user_id=user_id, request_id=seq,
                  started_s=round(started-start, 4), status='error', http_status=None,
                  ttft_s=None, latency_s=None, prompt_tokens=None, cached_prompt_tokens=None,
                  output_tokens=None, load_s=None, prefill_s=None, decode_s=None,
                  server_total_s=None, text_chars=0, error=None)
    payload = {'model': args.model, 'prompt': make_prompt(user_id, seq, args.prompt_words),
               'stream': True, 'keep_alive': args.keep_alive,
               'options': {'num_ctx': args.context, 'num_predict': args.max_tokens,
                           'temperature': 0, 'seed': args.seed}}
    async with lock:
        event_trace.append({'t_s': round(started-start, 4), 'event': 'submitted', 'user_id': user_id, 'request_id': seq})
    first = None
    final = None
    try:
        async with session.post(args.base_url.rstrip('/') + '/api/generate', json=payload) as resp:
            result['http_status'] = resp.status
            if resp.status != 200:
                result['error'] = (await resp.text())[:200]
            else:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    try:
                        part = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Invalid streamed JSON: {exc}') from exc
                    if part.get('error'):
                        raise ValueError(str(part['error']))
                    chunk = part.get('response', '')
                    if chunk:
                        result['text_chars'] += len(chunk)
                        if first is None:
                            first = time.perf_counter()
                    if part.get('done'):
                        final = part
                if final is None:
                    raise ValueError('Stream closed without final done event')
                result['status'] = 'ok'
                for name, key in [('prompt_tokens', 'prompt_eval_count'),
                                  ('cached_prompt_tokens','prompt_eval_cached_count'),
                                  ('output_tokens','eval_count')]:
                    result[name] = final.get(key)
                for name, key in [('load_s','load_duration'),('prefill_s','prompt_eval_duration'),
                                  ('decode_s','eval_duration'),('server_total_s','total_duration')]:
                    value = final.get(key)
                    result[name] = value / 1e9 if isinstance(value, (float,int)) else None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:300]
    finally:
        ended = time.perf_counter()
        result['latency_s'] = round(ended-started, 6)
        result['ttft_s'] = round(first-started, 6) if first is not None else None
        result['finished_s'] = round(ended-start, 4)
        async with lock:
            event_trace.append({'t_s': round(ended-start,4), 'event': 'finished', 'user_id': user_id, 'request_id': seq})
    return result


async def warmup(session, args):
    payload = {'model': args.model, 'prompt':'Reply OK.', 'stream':False,
               'keep_alive': args.keep_alive, 'options':{'num_predict':8,'num_ctx':args.context}}
    async with session.post(args.base_url.rstrip('/')+'/api/generate', json=payload,
                            timeout=aiohttp.ClientTimeout(total=args.timeout)) as resp:
        resp.raise_for_status()
        await resp.read()


def summarize(level, rows, started_at, ended_at, samples):
    good = [r for r in rows if r['status']=='ok']
    elapsed = ended_at - started_at
    def val(key):
        return [float(r[key]) for r in good if isinstance(r.get(key), (float, int))]
    output_tokens = sum(r.get('output_tokens') or 0 for r in good)
    return {'concurrency': level, 'attempted': len(rows), 'completed': len(good),
            'failed': len(rows)-len(good), 'failure_rate': (len(rows)-len(good))/len(rows) if rows else 0,
            'elapsed_s': round(elapsed, 4), 'requests_per_s': len(good)/elapsed if elapsed else None,
            'output_tokens_per_s': output_tokens/elapsed if elapsed else None,
            'total_output_tokens': output_tokens, 'ttft_p50_s': percentile(val('ttft_s'),50),
            'ttft_p95_s': percentile(val('ttft_s'),95), 'latency_p50_s':percentile(val('latency_s'),50),
            'latency_p95_s':percentile(val('latency_s'),95),
            'latency_mean_s':avg(val('latency_s')),
            'mean_request_decode_tps':avg([r['output_tokens']/r['decode_s'] for r in good
                                           if isinstance(r.get('output_tokens'),int) and r.get('decode_s')]),
            'gpu_peak_used_mib':max([s['gpu_used_mib'] for s in samples if 'gpu_used_mib' in s],default=None),
            'gpu_mean_util_pct':avg([s['gpu_util_pct'] for s in samples if 'gpu_util_pct' in s]),
            'ollama_peak_vram_mib':max([s['ollama_vram_mib'] for s in samples if 'ollama_vram_mib' in s],default=None)}


def diagnostics(summaries):
    notes = []
    one = next((r for r in summaries if r['concurrency']==1), None)
    if one is None:
        return notes
    for r in summaries:
        if r['concurrency'] == 1:
            continue
        if r['failure_rate'] > 0:
            notes.append(f"C={r['concurrency']}: {r['failed']}/{r['attempted']} failures. Check timeouts, 503 overload, and Ollama logs.")
        if (one['latency_p95_s'] and r['latency_p95_s'] and
            r['latency_p95_s'] > 2*one['latency_p95_s']):
            notes.append(f"C={r['concurrency']}: p95 latency exceeds twice single-user baseline; interactive responsiveness has degraded.")
    if len(summaries)>1:
        final = summaries[-1]
        if one['output_tokens_per_s'] and final['output_tokens_per_s']:
            ratio = final['output_tokens_per_s'] / one['output_tokens_per_s']
            notes.append(f"Aggregate generated-token throughput at max concurrency is {ratio:.2f}x the single-user rate; assess alongside p95 and failures.")
    notes.append('The client cannot directly measure Ollama internal queue wait. TTFT includes queueing, prefill, and transport; queue interpretation is inferential.')
    notes.append('Concurrency denotes in-flight clients, not confirmed GPU parallel batches. Check OLLAMA_NUM_PARALLEL and server logs.')
    return notes


async def run(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout,connector=connector,trust_env=False) as session:
        try:
            async with session.get(args.base_url.rstrip('/')+'/api/version', timeout=aiohttp.ClientTimeout(total=5)) as r:
                r.raise_for_status()
                version = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise SystemExit(f'Cannot contact Ollama at {args.base_url}: {exc}')
        await warmup(session,args)
        all_rows=[]; summaries=[]; all_samples=[]; all_events=[]
        for level in args.concurrency:
            print(f'Running concurrency={level}, {args.requests_per_user} requests per virtual user',flush=True)
            origin = time.perf_counter()
            monitor = Monitor(args.base_url, args.gpu_index, args.sample_interval)
            task = asyncio.create_task(monitor.run(session,origin))
            lock=asyncio.Lock(); events=[]
            gate=asyncio.Event()
            async def worker(i):
                await gate.wait()
                responses=[]
                for j in range(args.requests_per_user):
                    responses.append(await request_one(session,args,i,j,level,origin,events,lock))
                    if args.think_time:
                        await asyncio.sleep(args.think_time)
                return responses
            workers=[asyncio.create_task(worker(i)) for i in range(level)]
            started=time.perf_counter()
            gate.set()
            groups=await asyncio.gather(*workers)
            finished=time.perf_counter()
            monitor.stop.set(); await task
            rows=[item for group in groups for item in group]
            for row in rows:
                row['phase']=f'C{level}'
            for sample in monitor.samples:
                sample['concurrency']=level
            for ev in events:
                ev['concurrency']=level
            summary=summarize(level,rows,started,finished,monitor.samples)
            print('  ', {k:round(v,3) if isinstance(v,float) else v for k,v in summary.items() if k in ('attempted','failed','requests_per_s','latency_p95_s','ttft_p95_s','output_tokens_per_s','gpu_peak_used_mib')},flush=True)
            summaries.append(summary);all_rows.extend(rows);all_samples.extend(monitor.samples);all_events.extend(events)
            if args.cooldown:
                await asyncio.sleep(args.cooldown)
    meta={'created_utc':datetime.now(timezone.utc).isoformat(), 'model':args.model,
          'ollama_version':version, 'platform':platform.platform(),
          'parameters':{k:v for k,v in vars(args).items() if k != 'out'},
          'workload':'closed-loop fixed number of virtual users; per-user requests sequential; all users begin together',
          'metrics':'TTFT is first nonempty streamed text; p95 nearest rank; GPU samples device-wide; no internal queue timestamps'}
    root=Path(args.out)
    def dump(name,value): (root/name).write_text(json.dumps(value,indent=2)+'\n')
    dump('metadata.json',meta);dump('summary.json',{'levels':summaries,'diagnostics':diagnostics(summaries)})
    dump('runs.json',all_rows); dump('gpu_samples.json',all_samples);dump('events.json',all_events)
    if all_rows:
        with (root/'runs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    if summaries:
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print('Saved to',root)


def main():
    p=argparse.ArgumentParser(description='Local Ollama concurrency and saturation benchmark')
    p.add_argument('--model',default='qwen2.5:3b')
    p.add_argument('--base-url',default='http://127.0.0.1:11434')
    p.add_argument('--concurrency',nargs='+',type=int,default=LEVELS)
    p.add_argument('--requests-per-user',type=int,default=5)
    p.add_argument('--prompt-words',type=int,default=140)
    p.add_argument('--max-tokens',type=int,default=96)
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--keep-alive',default='20m')
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--sample-interval',type=float,default=0.6)
    p.add_argument('--gpu-index',type=int,default=0)
    p.add_argument('--cooldown',type=float,default=2)
    p.add_argument('--think-time',type=float,default=0)
    p.add_argument('--out',default='results/sweep')
    a=p.parse_args()
    if not a.concurrency or any(x<1 or x>64 for x in a.concurrency) or a.requests_per_user<1 or a.timeout<=0 or a.sample_interval<=0 or a.context<256 or a.max_tokens<1:
        p.error('Invalid benchmark settings')
    asyncio.run(run(a))

if __name__=='__main__': main()
