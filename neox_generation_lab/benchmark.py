import argparse,json,platform,random,statistics,time
from pathlib import Path
import torch
from model import TinyNeoX,load,cache_bytes
from generation import generate

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    p.add_argument('--prompt-tokens',type=int,default=64);p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--out',default='results/benchmark.json');a=p.parse_args()
    if min(a.prompt_tokens,a.tokens,a.repeats)<1:p.error('Counts must be positive')
    torch.set_num_threads(2);torch.manual_seed(42)
    m=load(a.checkpoint,a.device) if a.checkpoint else TinyNeoX().to(a.device).eval()
    if a.prompt_tokens+a.tokens>m.config.context:p.error('Context exceeded')
    prompt=[65+i%26 for i in range(a.prompt_tokens)]
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
        if a.device=='mps':torch.mps.synchronize()
    def trial(cached):
        marks=[]
        def emit(token):sync();marks.append(time.perf_counter())
        sync();start=time.perf_counter()
        out,trace=generate(m,prompt,a.tokens,cached,eos_id=None,on_token=emit)
        sync();end=time.perf_counter()
        return dict(total_ms=(end-start)*1000,first_token_ms=(marks[0]-start)*1000,
                    decode_mean_ms=(marks[-1]-marks[0])*1000/(len(marks)-1) if len(marks)>1 else None,
                    input_positions=sum(r['input_tokens'] for r in trace),final_cache_bytes=trace[-1]['cache_bytes']),out
    expected=trial(False)[1];assert trial(True)[1]==expected
    samples={False:[],True:[]};rng=random.Random(8)
    for _ in range(a.repeats):
        order=[False,True];rng.shuffle(order)
        for cached in order:
            result,out=trial(cached);assert out==expected;samples[cached].append(result)
    rows=[]
    for cached in (False,True):
        vals=samples[cached];rows.append(dict(cached=cached,median_total_ms=statistics.median(x['total_ms'] for x in vals),
            median_first_token_ms=statistics.median(x['first_token_ms'] for x in vals),
            median_decode_mean_ms=statistics.median(x['decode_mean_ms'] for x in vals) if a.tokens>1 else None,raw=vals))
    result=dict(arguments=vars(a),python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
                threads=2,dtype='float32',parameters=sum(x.numel() for x in m.parameters()),results=rows,
                scope='Batch one; fixed token count; random or user-loaded weights. Includes tensor creation, greedy selection, tracing, per-token synchronization. Excludes load, tokenization, printing. Cache bytes are logical tensor bytes, not peak memory.')
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
