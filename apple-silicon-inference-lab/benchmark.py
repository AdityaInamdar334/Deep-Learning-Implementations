#!/usr/bin/env python3
"""Client-observed inference benchmark. No third-party Python dependencies."""
import argparse,csv,json,os,platform,subprocess,time,urllib.request,urllib.error,statistics,pathlib,hashlib

ENGINES=('ollama','llama','mlx')

def post_stream(url,payload,engine,timeout=300):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    start=time.perf_counter(); first=None; text=[]; final={}; usage={}; errors=[]
    with urllib.request.urlopen(req,timeout=timeout) as res:
        for raw in res:
            line=raw.decode('utf-8','replace').strip()
            if not line: continue
            if engine=='ollama':
                try: msg=json.loads(line)
                except json.JSONDecodeError: continue
                piece=msg.get('response') or ''
                if piece:
                    if first is None: first=time.perf_counter()-start
                    text.append(piece)
                if msg.get('done'): final=msg
            else:
                if not line.startswith('data:'): continue
                data=line[5:].strip()
                if data=='[DONE]': continue
                try: msg=json.loads(data)
                except json.JSONDecodeError: continue
                if engine=='llama':
                    # llama-server exposes OpenAI-compatible completions here; usage may be final.
                    pass
                usage=msg.get('usage') or usage
                for choice in msg.get('choices',[]):
                    piece=choice.get('text') or (choice.get('delta') or {}).get('content') or ''
                    if piece:
                        if first is None: first=time.perf_counter()-start
                        text.append(piece)
                final=msg
    wall=time.perf_counter()-start
    out={'wall_s':round(wall,5),'ttft_visible_s':round(first,5) if first is not None else None,'text':''.join(text), 'chars':sum(map(len,text))}
    if engine=='ollama':
        prompt_n=final.get('prompt_eval_count');cached=final.get('prompt_eval_cached_count') or 0;eval_n=final.get('eval_count')
        pp=final.get('prompt_eval_duration');tg=final.get('eval_duration')
        out.update(prompt_tokens=prompt_n,completion_tokens=eval_n,cache_tokens=cached,
                   prefill_tps=round((prompt_n-cached)/(pp/1e9),3) if prompt_n is not None and pp and prompt_n>=cached else None,
                   decode_tps=round(eval_n/(tg/1e9),3) if eval_n is not None and tg else None,
                   load_s=round(final.get('load_duration',0)/1e9,4))
    else:
        out.update(prompt_tokens=usage.get('prompt_tokens'),completion_tokens=usage.get('completion_tokens'),
                   prefill_tps=None,decode_tps=None,load_s=None)
        # For OpenAI SSE servers usage is optional, so we do not invent tokens/sec.
    return out

def request_config(engine,model,prompt,max_tokens,seed,ctx):
    if engine=='ollama':
        return 'http://127.0.0.1:11434/api/generate',{'model':model,'prompt':prompt,'raw':True,'stream':True,'keep_alive':'10m','options':{'num_ctx':ctx,'num_predict':max_tokens,'temperature':0,'seed':seed,'top_k':1}}
    port=8080 if engine=='llama' else 8081
    return f'http://127.0.0.1:{port}/v1/completions',{'model':model,'prompt':prompt,'stream':True,'max_tokens':max_tokens,'temperature':0,'seed':seed}

def proc_rss_mib(pid):
    if not pid:return None
    try:
        r=subprocess.run(['ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=3)
        return round(int(r.stdout.strip())/1024,2)
    except (OSError,ValueError,subprocess.TimeoutExpired): return None

def ollama_ps():
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/ps',timeout=4) as r:
            return json.load(r).get('models',[])
    except (OSError,ValueError): return []

def run(args):
    root=pathlib.Path(args.out);root.mkdir(parents=True,exist_ok=True)
    p=pathlib.Path(args.prompt_file).read_text() if args.prompt_file else 'Summarize the relationship between KV cache, memory bandwidth, and local LLM generation in four clear sentences.'
    metadata={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'config':vars(args),'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(), 'note':'MLX four-bit and GGUF Q4_K_M have different quantization; memory fields are not comparable GPU allocations.'}
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for engine in args.engines:
        model={'ollama':args.ollama_model,'llama':args.llama_model,'mlx':args.mlx_model}[engine]
        url,payload=request_config(engine,model,p,args.max_tokens,args.seed,args.context)
        print(f'\n{engine} ({model})')
        for trial in range(-args.warmups,args.trials):
            warm=trial<0
            try:
                rss_start=proc_rss_mib(args.llama_pid if engine=='llama' else args.mlx_pid if engine=='mlx' else args.ollama_pid)
                result=post_stream(url,payload,engine,args.timeout)
                rss_end=proc_rss_mib(args.llama_pid if engine=='llama' else args.mlx_pid if engine=='mlx' else args.ollama_pid)
                row={'engine':engine,'model':model,'trial':trial,'warmup':warm,'status':'ok','process_rss_before_mib':rss_start,'process_rss_after_mib':rss_end,**result}
                if engine=='ollama':
                    models=ollama_ps();entry=next((m for m in models if model in (m.get('name'),m.get('model'))),{})
                    row['ollama_size_vram_bytes']=entry.get('size_vram');row['ollama_context_length']=entry.get('context_length')
            except Exception as e:
                row={'engine':engine,'model':model,'trial':trial,'warmup':warm,'status':'error','error':f'{type(e).__name__}: {e}'}
            if not warm:rows.append(row)
            print(('warmup' if warm else f'trial {trial+1}'),row['status'],'TTFT',row.get('ttft_visible_s'),'wall',row.get('wall_s'))
            (root/'runs.json').write_text(json.dumps(rows,indent=2))
    keys=sorted({k for r in rows for k in r})
    with (root/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary={}
    for engine in args.engines:
        data=[r for r in rows if r['engine']==engine and r['status']=='ok']
        summary[engine]={'successes':len(data),'failures':sum(r['engine']==engine and r['status']=='error' for r in rows)}
        for key in ('wall_s','ttft_visible_s','prefill_tps','decode_tps','process_rss_after_mib'):
            values=[r[key] for r in data if isinstance(r.get(key),(int,float))]
            summary[engine][f'median_{key}']=round(statistics.median(values),3) if values else None
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nSummary:',json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engines',nargs='+',choices=ENGINES,default=list(ENGINES))
    p.add_argument('--ollama-model',default='apple-duel');p.add_argument('--llama-model',default='apple-duel');p.add_argument('--mlx-model',default='mlx-community/Qwen2.5-3B-Instruct-4bit')
    p.add_argument('--prompt-file');p.add_argument('--trials',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--context',type=int,default=4096);p.add_argument('--max-tokens',type=int,default=128);p.add_argument('--seed',type=int,default=42);p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--ollama-pid',type=int);p.add_argument('--llama-pid',type=int);p.add_argument('--mlx-pid',type=int)
    p.add_argument('--out',default='results/duel')
    a=p.parse_args()
    if a.trials<1 or a.warmups<0: p.error('trials must be positive and warmups nonnegative')
    run(a)
if __name__=='__main__':main()
