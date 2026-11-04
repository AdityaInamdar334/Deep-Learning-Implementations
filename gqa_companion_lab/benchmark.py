"""End-to-end prefill + teacher-forced decode timing. Not a serving benchmark."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from model import Config,TinyGQA,cache_bytes
from train import device_for

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps','auto'])
    p.add_argument('--backend',default='repeat',choices=['repeat','grouped','native'])
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--prompt',type=int,default=64)
    p.add_argument('--decode',type=int,default=32)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',default='results/benchmark.json')
    a=p.parse_args()
    if min(a.batch,a.prompt,a.decode,a.repeats)<1 or a.prompt+a.decode>256:p.error('Invalid sequence dimensions')
    torch.set_num_threads(2);torch.manual_seed(42)
    device=device_for(a.device)
    ids=torch.randint(0,256,(a.batch,a.prompt+a.decode),device=device)
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
        if device=='mps':torch.mps.synchronize()
    rows=[]
    for kv in (8,2,1):
        model=TinyGQA(Config(num_kv_heads=kv,backend=a.backend)).to(device).eval()
        @torch.inference_mode()
        def trial():
            sync();t0=time.perf_counter()
            _,cache=model(ids[:,:a.prompt],use_cache=True)
            sync();t1=time.perf_counter()
            for index in range(a.prompt,a.prompt+a.decode):
                _,cache=model(ids[:,index:index+1],cache,use_cache=True)
            sync();t2=time.perf_counter()
            return (t1-t0)*1000,(t2-t1)*1000/a.decode,cache_bytes(cache)
        try:
            trial()  # Warm up kernels and allocation paths before recording.
            measurements=[trial() for _ in range(a.repeats)]
        except (RuntimeError,NotImplementedError) as exc:
            raise RuntimeError(f'Backend {a.backend} failed; choose another explicitly') from exc
        row=dict(kv_heads=kv,prefill_ms_median=statistics.median(x[0] for x in measurements),
                 decode_step_ms_median=statistics.median(x[1] for x in measurements),
                 final_cache_bytes=measurements[-1][2],raw_measurements=measurements)
        print(json.dumps(row),flush=True);rows.append(row)
        del model
    result=dict(arguments=vars(a),device=device,torch_version=str(torch.__version__),
                dtype='float32',cpu_threads=2,results=rows,
                scope='Includes full model forward, logits, masks, and torch.cat cache growth; no tokenizer or sampling')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
