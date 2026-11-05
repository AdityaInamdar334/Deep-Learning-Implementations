import argparse,json
from pathlib import Path
import torch
from model import load,TinyNeoX
from generation import generate,encode,ByteStream

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--prompt',default='The cache ')
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--temperature',type=float,default=0)
    p.add_argument('--top-k',type=int);p.add_argument('--no-cache',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');p.add_argument('--trace',default='results/trace.json')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    m=load(a.checkpoint,a.device) if a.checkpoint else TinyNeoX().to(a.device).eval()
    if not a.checkpoint:print('Random weights: mechanics demonstration, not meaningful language.')
    stream=ByteStream();print(a.prompt,end='',flush=True)
    def emit(token):print(stream.push(token),end='',flush=True)
    _,trace=generate(m,encode(a.prompt),a.tokens,not a.no_cache,a.temperature,a.top_k,on_token=emit)
    print(stream.finish());out=Path(a.trace);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(trace,indent=2))
if __name__=='__main__':main()
