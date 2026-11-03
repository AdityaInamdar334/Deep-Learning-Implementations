"""Optional checkpoint comparator. Reads safetensors only (not pickle).

Usage: pip install safetensors
python audit_safetensors.py old.safetensors new.safetensors
A single-file utility; sharded checkpoints require comparing shards separately.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    from safetensors import safe_open
    parser=argparse.ArgumentParser();parser.add_argument('before');parser.add_argument('after');a=parser.parse_args()
    for f in [a.before,a.after]:
        if not Path(f).is_file():raise FileNotFoundError(f)
    with safe_open(a.before,framework='pt',device='cpu') as old, safe_open(a.after,framework='pt',device='cpu') as new:
        if set(old.keys()) != set(new.keys()):raise ValueError('tensor keys differ')
        changed=total=0;per_tensor={}
        for k in sorted(old.keys()):
            x=old.get_tensor(k);y=new.get_tensor(k)
            if x.shape != y.shape or x.dtype != y.dtype:raise ValueError(f'shape or dtype differs: {k}')
            n=int((x!=y).sum().item()); per_tensor[k]=n
            changed+=n; total+=x.numel()
    print(json.dumps({'changed_elements':changed,'total_elements':total,'fraction':changed/total,'tensors_changed':{k:v for k,v in per_tensor.items() if v}},indent=2))
if __name__=='__main__':main()
