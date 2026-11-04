import argparse,json
import torch
from model import Config,TinyGQA,cache_bytes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    if a.batch<1 or not 1<=a.length<=256: p.error('Invalid batch or length')
    torch.set_num_threads(2)
    reports=[]
    for kv in (8,2,1):
        c=Config(num_kv_heads=kv)
        m=TinyGQA(c).eval()
        with torch.no_grad():
            _,cache=m(torch.zeros(a.batch,a.length,dtype=torch.long),use_cache=True)
        actual=cache_bytes(cache)
        expected=2*c.num_layers*a.batch*a.length*kv*(c.hidden_size//c.num_heads)*4
        assert actual==expected
        row=dict(kv_heads=kv,query_heads=c.num_heads,cache_shape=list(cache[0][0].shape),
                 cache_bytes=actual,cache_MiB=actual/2**20,
                 parameters=sum(p.numel() for p in m.parameters()))
        reports.append(row)
        print(json.dumps(row))
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(reports,indent=2))
if __name__=='__main__':main()
