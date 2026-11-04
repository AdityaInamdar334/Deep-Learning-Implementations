"""Step-by-step causal dependency demonstration, no model training needed."""
import torch
from .models import causal_shift, AverageShortener, upsample_repeat

def main():
    t = torch.arange(1,13,dtype=torch.float32).view(1,12,1)
    k = 3
    for label,shift in [('unsafe, no shift',0),('safe, shift k-1',k-1)]:
        data = causal_shift(t,shift)
        pooled = AverageShortener()(data,k)
        restored = upsample_repeat(pooled,k,t.size(1))
        changed=t.clone();changed[0,2,0]=999
        changed_restored=upsample_repeat(AverageShortener()(causal_shift(changed,shift),k),k,t.size(1))
        print(label)
        print(' input:',t.flatten().tolist())
        print(' shifted:',data.flatten().tolist())
        print(' pooled:',[round(v,2) for v in pooled.flatten().tolist()])
        print(' change token index 2 affects output indices:',
              (restored!=changed_restored).nonzero()[:,1].unique().tolist())
        print()

if __name__=='__main__': main()
