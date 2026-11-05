"""Analytical logical KV bytes. Excludes weights, workspace and allocator overhead."""
import argparse,json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,default=44);p.add_argument('--heads',type=int,default=64)
    p.add_argument('--head-dim',type=int,default=96);p.add_argument('--tokens',type=int,default=2048)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--bytes-per-element',type=int,default=2);a=p.parse_args()
    if min(vars(a).values())<1:p.error('Values must be positive')
    n=2*a.layers*a.heads*a.head_dim*a.tokens*a.batch*a.bytes_per_element
    print(json.dumps(dict(bytes=n,GiB=n/2**30,MiB=n/2**20,assumptions=vars(a)),indent=2))
