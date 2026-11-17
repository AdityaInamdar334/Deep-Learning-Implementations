#!/usr/bin/env python3
"""Reproducible local Q4/Q8/F16 comparison using Ollama raw-completion endpoint.
Standard library only. The harness grades outcomes, not model 'intelligence'.
"""
import argparse, csv, hashlib, json, math, os, platform, re, shutil, statistics, subprocess, threading, time, urllib.request, urllib.error
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SYSTEM='You are a precise assistant. Follow the user instruction, return only the requested answer, and do not add explanations unless asked.'
FIELDS=['model','task_id','category','repeat','score','passed','response','expected','grader','prompt_tokens','cached_prompt_tokens','output_tokens','prefill_tps','decode_tps','wall_s','load_s','peak_gpu_mib','baseline_gpu_mib','incremental_gpu_mib','ollama_vram_bytes','error']

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def request_json(url,payload=None,timeout=240):
 body=json.dumps(payload).encode() if payload is not None else None
 req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'} if body is not None else {},method='POST' if body is not None else 'GET')
 with urllib.request.urlopen(req,timeout=timeout) as r: return json.load(r)

def gpu_mib(index=0):
 if not shutil.which('nvidia-smi'): return None
 try:
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(index)],capture_output=True,text=True,timeout=4,check=True)
  return float(p.stdout.strip().splitlines()[0])
 except (ValueError,IndexError,OSError,subprocess.SubprocessError): return None

class Sampler:
 def __init__(self,idx=0,dt=0.15):self.idx,self.dt,self.samples,self.stop=idx,dt,[],threading.Event();self.thread=None
 def start(self):
  def loop():
   while not self.stop.is_set():
    x=gpu_mib(self.idx)
    if x is not None:self.samples.append(x)
    self.stop.wait(self.dt)
  self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
 def finish(self):
  self.stop.set()
  if self.thread:self.thread.join(timeout=5)
  return max(self.samples) if self.samples else None

def grade(task,response):
 kind=task['grader']; value=task['expected']; s=response.strip()
 if kind=='exact':return int(s==str(value))
 if kind=='integer':
  cleaned=s.replace(',','').replace(' ','')
  return int(bool(re.fullmatch(r'[+-]?\d+',cleaned)) and int(cleaned)==int(value))
 if kind=='json':
  try:
   # No markdown fences allowed: strict schema and values.
   parsed=json.loads(s)
   return int(isinstance(parsed,dict) and set(parsed)==set(value) and all(type(parsed[k]) is type(v) and parsed[k]==v for k,v in value.items()))
  except (ValueError,TypeError):return 0
 if kind=='set':
  try:
   parsed=json.loads(s)
   return int(isinstance(parsed,list) and len(parsed)==len(value) and set(map(str,parsed))==set(map(str,value)))
  except (ValueError,TypeError):return 0
 raise ValueError('Unknown grader: '+kind)

def model_vram(base,model):
 try:
  data=request_json(base+'/api/ps',timeout=8)
  for item in data.get('models',[]):
   if item.get('name','').split(':')[0]==model.split(':')[0] and (item.get('name')==model or item.get('model')==model):return item.get('size_vram')
 except (OSError,ValueError,urllib.error.URLError):pass
 return None

def run_one(base,task,model,trial,ctx,outmax,gpu_index):
 prompt=SYSTEM+'\n\nUSER:\n'+task['prompt']+'\n\nASSISTANT:\n'
 payload={'model':model,'prompt':prompt,'raw':True,'stream':False,'keep_alive':'0s','options':{'num_ctx':ctx,'num_predict':outmax,'temperature':0,'seed':42,'top_k':1,'repeat_penalty':1.0}}
 baseline=gpu_mib(gpu_index);sampler=Sampler(gpu_index);sampler.start();start=time.perf_counter();error='';data={};vram=None
 try:
  # keep_alive 0s unloads immediately so /api/ps may not show residency; ask for 2 minutes then explicitly unload below.
  payload['keep_alive']='2m'
  data=request_json(base+'/api/generate',payload,timeout=600)
  vram=model_vram(base,model)
 except Exception as ex:error=f'{type(ex).__name__}: {ex}'
 elapsed=time.perf_counter()-start;peak=sampler.finish()
 try:request_json(base+'/api/generate',{'model':model,'keep_alive':0},timeout=90)
 except Exception:pass
 response=data.get('response','');passed=grade(task,response) if not error else 0
 pc=int(data.get('prompt_eval_count') or 0); cached=int(data.get('prompt_eval_cached_count') or 0); out=int(data.get('eval_count') or 0)
 pe=float(data.get('prompt_eval_duration') or 0);de=float(data.get('eval_duration') or 0)
 return {'model':model,'task_id':task['id'],'category':task['category'],'repeat':trial,'score':passed,'passed':passed,'response':response,'expected':json.dumps(task['expected'],ensure_ascii=False),'grader':task['grader'],'prompt_tokens':pc,'cached_prompt_tokens':cached,'output_tokens':out,'prefill_tps':round(max(0,pc-cached)*1e9/pe,2) if pe>0 else None,'decode_tps':round(out*1e9/de,2) if de>0 else None,'wall_s':round(elapsed,3),'load_s':round(float(data.get('load_duration') or 0)/1e9,3),'peak_gpu_mib':peak,'baseline_gpu_mib':baseline,'incremental_gpu_mib':round(max(0,peak-baseline),2) if peak is not None and baseline is not None else None,'ollama_vram_bytes':vram,'error':error}

def summaries(rows):
 summary=[]
 for model in sorted(set(x['model'] for x in rows)):
  for cat in sorted(set(x['category'] for x in rows)):
   a=[x for x in rows if x['model']==model and x['category']==cat];successful=[x for x in a if not x['error']]
   def median(key):
    v=[r[key] for r in successful if r[key] is not None];return round(statistics.median(v),3) if v else None
   summary.append({'model':model,'category':cat,'n':len(a),'completed':len(successful),'accuracy':round(sum(r['score'] for r in a)/len(a),4) if a else None,'median_decode_tps':median('decode_tps'),'median_prefill_tps':median('prefill_tps'),'median_wall_s':median('wall_s'),'max_gpu_mib':max((r['peak_gpu_mib'] for r in successful if r['peak_gpu_mib'] is not None),default=None)})
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['quant-q4','quant-q8','quant-f16']);p.add_argument('--trials',type=int,default=3);p.add_argument('--ctx',type=int,default=4096);p.add_argument('--max-tokens',type=int,default=160);p.add_argument('--gpu-index',type=int,default=0);p.add_argument('--base',default='http://127.0.0.1:11434');p.add_argument('--dataset',default=str(ROOT/'data'/'tasks.json'));p.add_argument('--out',default=str(ROOT/'results'/'run'));p.add_argument('--models-dir',default=str(ROOT/'models'));args=p.parse_args()
 tasks=json.loads(Path(args.dataset).read_text());assert isinstance(tasks,list) and all(set(['id','category','prompt','grader','expected'])<=set(t) for t in tasks)
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 files={}
 for key in ('q4','q8','f16'):
  candidates=list(Path(args.models_dir).glob(f'*{key}*.gguf'))
  for path in candidates:files[path.name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
 meta={'platform':platform.platform(),'python':platform.python_version(),'models':args.models,'trials':args.trials,'context':args.ctx,'max_tokens':args.max_tokens,'dataset_sha256':sha256(args.dataset),'gguf_files':files,'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'ollama_version':None}
 try:meta['ollama_version']=request_json(args.base+'/api/version',timeout=6)
 except Exception:pass
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 rows=[]
 with (out/'runs.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=FIELDS);w.writeheader()
  for trial in range(1,args.trials+1):
   for task in tasks:
    # Rotate model order by trial to mitigate systematic drift.
    models=args.models[(trial-1)%len(args.models):]+args.models[:(trial-1)%len(args.models)]
    for model in models:
     print(f'[{trial}/{args.trials}] {task["id"]}: {model}',flush=True)
     row=run_one(args.base,task,model,trial,args.ctx,args.max_tokens,args.gpu_index);rows.append(row);w.writerow(row);fh.flush()
     if row['error']:print('  ERROR',row['error'],flush=True)
 (out/'summary.json').write_text(json.dumps(summaries(rows),indent=2))
 print('Saved',out/'runs.csv',out/'summary.json')

if __name__=='__main__':main()
