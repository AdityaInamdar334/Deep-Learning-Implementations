#!/usr/bin/env python3
"""Local inference diagnostic CLI. Python 3.10+, standard library only."""
import argparse, csv, json, math, platform, shutil, statistics, subprocess, threading, time, urllib.request, urllib.error
from pathlib import Path

DEFAULT_PROMPT = ('Explain the difference between prompt prefill and autoregressive decoding. '
                  'Give a concrete example involving a local programming assistant.')

def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def get_placement(base, model):
    try:
        data=get_json(base.rstrip('/')+'/api/ps', timeout=5)
        for x in data.get('models', []):
            name=x.get('name','')
            if name==model or name.split(':')[0]==model.split(':')[0]:
                return {'name':name,'size_bytes':x.get('size'),'size_vram_bytes':x.get('size_vram'),
                        'context_length':x.get('context_length'), 'details':x.get('details',{})}
    except Exception: pass
    return None

def get_ollama_ps():
    if not shutil.which('ollama'): return None
    try:
        p=subprocess.run(['ollama','ps'],capture_output=True,text=True,timeout=6)
        return p.stdout.strip() if p.returncode==0 else None
    except Exception: return None

def gpu_sample(gpu_index):
    if not shutil.which('nvidia-smi'): return None
    try:
        p=subprocess.run(['nvidia-smi','--id='+str(gpu_index),
                          '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if p.returncode: return None
        parts=[v.strip() for v in p.stdout.strip().splitlines()[0].split(',')]
        vals=[]
        for x in parts:
            try: vals.append(float(x))
            except ValueError: vals.append(None)
        return dict(zip(['used_mib','total_mib','gpu_util_pct','memory_util_pct','temp_c','power_w'],vals))
    except Exception: return None

class Sampler:
    def __init__(self,gpu_index,interval):
        self.gpu_index=gpu_index; self.interval=interval; self.samples=[]; self.stop_event=threading.Event(); self.thread=None
    def start(self):
        def work():
            while not self.stop_event.is_set():
                sample=gpu_sample(self.gpu_index)
                if sample is not None: self.samples.append({'offset_s':time.perf_counter()-self.t0,**sample})
                self.stop_event.wait(self.interval)
        self.t0=time.perf_counter(); self.thread=threading.Thread(target=work,daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=7)

def ns_s(v):
    return v/1e9 if isinstance(v,(int,float)) else None

def safe_ratio(a,b):
    return a/b if isinstance(a,(int,float)) and isinstance(b,(int,float)) and b>0 else None

def stream_generate(base, model, prompt, context, max_tokens, timeout, seed, raw=True):
    payload={'model':model,'prompt':prompt,'stream':True,'raw':raw,'keep_alive':'5m',
             'options':{'num_ctx':context,'num_predict':max_tokens,'temperature':0,'seed':seed}}
    req=urllib.request.Request(base.rstrip('/')+'/api/generate',data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'},method='POST')
    start=time.perf_counter(); first_event=None; first_text=None; chunks=0; final={}; outputs=[]
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            arrival=time.perf_counter()
            if not line.strip(): continue
            obj=json.loads(line)
            if first_event is None: first_event=arrival-start
            s=obj.get('response','')
            if s:
                chunks+=1; outputs.append(s)
                if first_text is None: first_text=arrival-start
            if obj.get('done'):
                final=obj
                break
    wall=time.perf_counter()-start
    if not final.get('done'):
        raise RuntimeError('Stream ended without a final done=true metrics record')
    prompt_tokens=final.get('prompt_eval_count'); cached=final.get('prompt_eval_cached_count',0) or 0
    prefill_s=ns_s(final.get('prompt_eval_duration')); decode_s=ns_s(final.get('eval_duration'))
    generated=final.get('eval_count')
    return {'model':model,'context':context,'max_tokens':max_tokens,'prompt_chars':len(prompt),
            'client_ttft_s':first_text,'client_first_event_s':first_event, 'wall_s':wall,
            'load_s':ns_s(final.get('load_duration')),'server_total_s':ns_s(final.get('total_duration')),
            'server_prefill_s':prefill_s,'server_decode_s':decode_s,
            'prompt_tokens':prompt_tokens,'cached_prompt_tokens':cached,'generated_tokens':generated,
            'prefill_tps':safe_ratio((prompt_tokens or 0)-cached,prefill_s),
            'decode_tps':safe_ratio(generated,decode_s),
            'chunks':chunks,'done_reason':final.get('done_reason'), 'output':''.join(outputs)}

def diagnose(r, placement):
    flags=[]; notes=[]
    if r.get('client_ttft_s') is None:
        flags.append('NO_VISIBLE_TOKEN'); notes.append('No nonempty streamed text was observed; client TTFT is unavailable.')
    if r.get('error'):
        flags.append('REQUEST_ERROR'); notes.append('The inference request failed; inspect the error and server logs first.')
        return flags,notes
    if placement and placement.get('size_bytes') and placement.get('size_vram_bytes') is not None:
        total=placement['size_bytes']; vram=placement['size_vram_bytes']; fraction=vram/total if total else None
        if fraction is not None and fraction < 0.95:
            flags.append('POSSIBLE_CPU_OFFLOAD'); notes.append(f'Ollama reports {fraction:.0%} of model allocation in VRAM. Confirm placement using `ollama ps`; split placement may slow inference.')
    if r.get('load_s') is not None and r.get('wall_s') and r['load_s']>=0.25*r['wall_s'] and r['load_s']>0.5:
        flags.append('LOAD_DOMINATED'); notes.append('Model loading accounts for a substantial fraction of request wall time; test a warm, resident model.')
    if r.get('server_prefill_s') and r.get('server_decode_s'):
        pf=r['server_prefill_s']; dec=r['server_decode_s']
        if pf>dec*1.5 and (r.get('prompt_tokens') or 0)>250:
            flags.append('PREFILL_DOMINATED'); notes.append('Server reports more time evaluating input than generating output; compare shorter prompts at fixed output length.')
        elif dec>pf*2 and (r.get('generated_tokens') or 0)>=24:
            flags.append('DECODE_DOMINATED'); notes.append('Server reports most model execution time in decoding; compare quantization, GPU residency, and longer output tests.')
    samples=r.get('gpu_samples') or []
    if samples:
        util=[x['gpu_util_pct'] for x in samples if x.get('gpu_util_pct') is not None]
        peak=max((x['used_mib'] for x in samples if x.get('used_mib') is not None),default=None)
        total=next((x['total_mib'] for x in samples if x.get('total_mib')),None)
        if peak and total and peak/total>=0.9:
            flags.append('HIGH_VRAM_PRESSURE'); notes.append('Sampled device-wide VRAM usage exceeded 90%; this is a risk indicator, not proof of an OOM or offload.')
        if util and statistics.median(util)<25 and (r.get('decode_tps') or 1e6)<20:
            flags.append('LOW_GPU_UTILIZATION'); notes.append('Sparse GPU-utilization samples are low while decoding is slow; investigate placement, CPU activity, and GPU sampling limits before attributing cause.')
    if not flags: notes.append('No strong bottleneck signature from available measurements. Run controlled A/B prompts and inspect logs; utilization alone is not causal evidence.')
    return flags,notes

def fmt(v, digits=2): return f'{v:.{digits}f}' if isinstance(v,(int,float)) else 'n/a'

def main(argv=None):
    ap=argparse.ArgumentParser(description='Diagnose Ollama local inference with streaming timing and optional NVIDIA telemetry.')
    ap.add_argument('--model',default='qwen2.5:3b'); ap.add_argument('--base-url',default='http://127.0.0.1:11434')
    ap.add_argument('--prompt',default=DEFAULT_PROMPT); ap.add_argument('--prompt-file')
    ap.add_argument('--context',type=int,default=4096); ap.add_argument('--max-tokens',type=int,default=192)
    ap.add_argument('--runs',type=int,default=3); ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--gpu-index',type=int,default=0); ap.add_argument('--sample-interval',type=float,default=0.35)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--timeout',type=int,default=240)
    ap.add_argument('--out',default='results/diagnosis'); ap.add_argument('--no-gpu',action='store_true')
    a=ap.parse_args(argv)
    if a.context<128 or a.max_tokens<1 or a.runs<1 or a.warmup<0 or a.sample_interval<0.1: ap.error('Invalid context, token count, runs, or interval')
    prompt=Path(a.prompt_file).read_text(encoding='utf-8') if a.prompt_file else a.prompt
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    try: vers=get_json(a.base_url.rstrip('/')+'/api/version')
    except Exception as e: ap.error(f'Ollama API not reachable: {e}')
    print('Ollama:',vers,'Model:',a.model)
    print('Warmups:',a.warmup,'Measured runs:',a.runs)
    for i in range(a.warmup):
        try: stream_generate(a.base_url,a.model,prompt,a.context,a.max_tokens,a.timeout,a.seed)
        except Exception as e: ap.error(f'Warmup failed: {e}')
    rows=[]
    for i in range(a.runs):
        placement_before=get_placement(a.base_url,a.model); ps_text=get_ollama_ps()
        baseline=None if a.no_gpu else gpu_sample(a.gpu_index)
        sampler=Sampler(a.gpu_index,a.sample_interval) if not a.no_gpu else None
        if sampler: sampler.start()
        try:
            result=stream_generate(a.base_url,a.model,prompt,a.context,a.max_tokens,a.timeout,a.seed)
        except Exception as e:
            result={'model':a.model,'context':a.context,'error':str(e)}
        finally:
            if sampler: sampler.stop()
        placement_after=get_placement(a.base_url,a.model)
        result.update({'run':i+1,'gpu_baseline':baseline,'gpu_samples':sampler.samples if sampler else [],
                       'placement':placement_after or placement_before,'ollama_ps':get_ollama_ps() or ps_text})
        result['gpu_peak_used_mib']=max((s['used_mib'] for s in result['gpu_samples'] if s.get('used_mib') is not None),default=None)
        result['gpu_median_util_pct']=statistics.median([s['gpu_util_pct'] for s in result['gpu_samples'] if s.get('gpu_util_pct') is not None]) if any(s.get('gpu_util_pct') is not None for s in result['gpu_samples']) else None
        result['flags'],result['interpretation']=diagnose(result,result['placement'])
        rows.append(result)
        print(f"Run {i+1}: TTFT={fmt(result.get('client_ttft_s'))}s  prefill={fmt(result.get('prefill_tps'))}tok/s  decode={fmt(result.get('decode_tps'))}tok/s  GPU peak={fmt(result.get('gpu_peak_used_mib'),0)}MiB  flags={','.join(result['flags']) or 'none'}")
        (out/'runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    cols=['run','model','context','client_ttft_s','wall_s','load_s','server_prefill_s','server_decode_s', 'prompt_tokens','cached_prompt_tokens','generated_tokens','prefill_tps','decode_tps','gpu_peak_used_mib','gpu_median_util_pct','flags','error']
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader()
        for r in rows: writer.writerow({k:','.join(r.get(k,[])) if k=='flags' else r.get(k) for k in cols})
    metrics=['client_ttft_s','wall_s','load_s','prefill_tps','decode_tps','gpu_peak_used_mib','gpu_median_util_pct']
    summary={k:statistics.median([r[k] for r in rows if isinstance(r.get(k),(int,float))]) if any(isinstance(r.get(k),(int,float)) for r in rows) else None for k in metrics}
    summary['run_count']=len(rows); summary['success_count']=sum(not r.get('error') for r in rows)
    summary['flags_observed']=sorted(set(f for r in rows for f in r['flags']))
    metadata={'ollama_version':vers,'python':platform.python_version(),'platform':platform.platform(),
              'model':a.model,'context':a.context,'max_tokens':a.max_tokens,'gpu_index':a.gpu_index,
              'sample_interval':a.sample_interval,'warmup':a.warmup,'runs':a.runs,
              'disclaimer':'Client TTFT is first nonempty streamed text, affected by buffering and network; not the server prefill duration. GPU telemetry is device-wide and sampled.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nMedian summary:',json.dumps(summary,indent=2))
    print('Artifacts:',out.resolve())
    return 0 if summary['success_count']==len(rows) else 1

if __name__=='__main__': raise SystemExit(main())
