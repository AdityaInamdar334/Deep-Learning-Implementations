#!/usr/bin/env python3
"""Compare Ollama /api/generate raw with llama.cpp /completion using identical GGUF and prompt bytes."""
import argparse,csv,hashlib,json,platform,statistics,time,urllib.request,urllib.error
from pathlib import Path

def post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r: return json.load(r)

def run(engine,url,model,prompt,seed,max_tokens,timeout):
    if engine=='ollama':
        obj=post(url+'/api/generate',{'model':model,'prompt':prompt,'raw':True,'stream':False,'keep_alive':'30m','options':{'temperature':0,'seed':seed,'num_ctx':4096,'num_predict':max_tokens}},timeout)
        ns=1e9; pp=obj.get('prompt_eval_count'); pd=obj.get('prompt_eval_duration'); nt=obj.get('eval_count'); td=obj.get('eval_duration')
        return {'output':obj.get('response',''),'prompt_tokens':pp,'generated_tokens':nt,'prefill_tps':pp*ns/pd if pp and pd else None,'decode_tps':nt*ns/td if nt and td else None,'load_ms':(obj.get('load_duration') or 0)/1e6,'stop_reason':obj.get('done_reason'),'raw':obj}
    obj=post(url+'/completion',{'prompt':prompt,'stream':False,'cache_prompt':False,'temperature':0,'seed':seed,'n_predict':max_tokens},timeout)
    t=obj.get('timings') or {}; pp=t.get('prompt_n'); nt=t.get('predicted_n')
    # The server reports speed directly; no inference from request wall time.
    return {'output':obj.get('content',''),'prompt_tokens':pp,'generated_tokens':nt,'prefill_tps':t.get('prompt_per_second'),'decode_tps':t.get('predicted_per_second'),'load_ms':None,'stop_reason':obj.get('stop_type') or ('limit' if obj.get('stopped_limit') else 'other'),'raw':obj}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ollama-model',required=True,help='Ollama model created FROM the same local GGUF')
    ap.add_argument('--model-file',required=True)
    ap.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    ap.add_argument('--llama-url',default='http://127.0.0.1:8080')
    ap.add_argument('--trials',type=int,default=3)
    ap.add_argument('--max-tokens',type=int,default=128)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--timeout',type=int,default=240)
    ap.add_argument('--output-dir',default='results')
    args=ap.parse_args()
    if args.trials<1: ap.error('--trials must be >=1')
    model=Path(args.model_file).expanduser().resolve()
    if not model.is_file(): ap.error('GGUF file does not exist')
    sha=hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):sha.update(chunk)
    prompts=json.loads((Path(__file__).parent/'prompts'/'tasks.json').read_text())
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    metadata={'model_path':str(model),'model_sha256':sha.hexdigest(),'model_bytes':model.stat().st_size,'ollama_model':args.ollama_model,'python':platform.python_version(),'platform':platform.platform(),'seed':args.seed,'max_tokens':args.max_tokens,'trials':args.trials,'context':4096,'temperature':0,'mode':'raw completion; no chat template; llama.cpp cache_prompt false'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for trial in range(args.trials):
        for task in prompts:
            for engine,url in [('ollama',args.ollama_url),('llama.cpp',args.llama_url)]:
                start=time.perf_counter(); error=''
                try:
                    result=run(engine,url,args.ollama_model,task['prompt'],args.seed,args.max_tokens,args.timeout)
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}';result={}
                elapsed=time.perf_counter()-start
                output=result.get('output','')
                row={'trial':trial+1,'task_id':task['id'],'engine':engine,'wall_seconds':round(elapsed,4),'prefill_tps':result.get('prefill_tps'),'decode_tps':result.get('decode_tps'),'prompt_tokens':result.get('prompt_tokens'),'generated_tokens':result.get('generated_tokens'),'load_ms':result.get('load_ms'),'stop_reason':result.get('stop_reason'),'output':output,'expected_substring':task.get('expected_substring',''),'contains_expected':task.get('expected_substring','').lower() in output.lower() if not error and task.get('expected_substring') else None,'error':error}
                rows.append(row)
                print(f"{engine:9} trial={trial+1} {task['id']:12} prefill={str(row['prefill_tps']):>10} decode={str(row['decode_tps']):>10} elapsed={elapsed:.2f}s {'ERROR '+error if error else ''}")
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={}
    for engine in ['ollama','llama.cpp']:
        selected=[r for r in rows if r['engine']==engine and not r['error']]
        def med(key):
            vals=[float(r[key]) for r in selected if r[key] is not None];return round(statistics.median(vals),2) if vals else None
        summary[engine]={'successful_runs':len(selected),'failed_runs':sum(bool(r['error']) for r in rows if r['engine']==engine),'median_prefill_tps':med('prefill_tps'),'median_decode_tps':med('decode_tps'),'median_wall_seconds':med('wall_seconds'),'substring_checks_passed':sum(r['contains_expected'] is True for r in selected),'substring_checks_total':sum(r['contains_expected'] is not None for r in selected)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\n'+json.dumps(summary,indent=2))
    print('\nNote: substring checks are smoke tests, NOT a model quality benchmark.')
if __name__=='__main__':main()
