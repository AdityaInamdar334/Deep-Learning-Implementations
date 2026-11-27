import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(1)
ROOT=Path(__file__).parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
def batch(n,t,rng):
    tokens=torch.randint(0,2,(n,t),generator=rng)
    x=torch.nn.functional.one_hot(tokens,2).float()
    marker=torch.zeros(n,t,1); marker[:,0]=1
    return torch.cat((x,marker),-1),tokens[:,0]
class MemoryModel(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        if kind=='attention':
            self.key=nn.Linear(3,32); self.value=nn.Linear(3,32); self.query=nn.Parameter(torch.randn(32)*.1)
        else: self.core={'RNN':nn.RNN,'GRU':nn.GRU,'LSTM':nn.LSTM}[kind](3,32,batch_first=True)
        self.head=nn.Linear(32,2)
    def forward(self,x):
        if self.kind=='attention':
            a=(self.key(x)@self.query/32**.5).softmax(dim=1)
            h=(a.unsqueeze(-1)*self.value(x)).sum(1)
        else: h=self.core(x)[0][:,-1]
        return self.head(h)
def check_bptt():
    rng=np.random.default_rng(1); x=rng.normal(size=(7,3)); U=rng.normal(size=(4,3))*.2; W=rng.normal(size=(4,4))*.2; h=[np.zeros(4)]
    for a in x: h.append(np.tanh(U@a+W@h[-1]))
    d=h[-1].copy(); gW=np.zeros_like(W); gU=np.zeros_like(U)
    for t in reversed(range(len(x))):
        da=d*(1-h[t+1]**2); gW+=np.outer(da,h[t]); gU+=np.outer(da,x[t]); d=W.T@da
    u=torch.tensor(U,requires_grad=True); w=torch.tensor(W,requires_grad=True); z=torch.zeros(4,dtype=torch.float64)
    for a in torch.tensor(x): z=torch.tanh(u@a+w@z)
    (z.square().sum()/2).backward(); errors=[float(np.max(abs(gU-u.grad.numpy()))),float(np.max(abs(gW-w.grad.numpy())))]; assert max(errors)<1e-12; return errors
def main():
    errors=check_bptt(); steps=np.arange(101)
    for w in [.9,1.,1.1]: plt.semilogy(steps,w**steps,label=f'w={w}')
    plt.xlabel('Distance in time'); plt.ylabel('Absolute derivative'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'gradient.png',dpi=180); plt.close()
    r=[]
    for seed in [11,22,33]:
        for kind in ['RNN','GRU','LSTM','attention']:
            torch.manual_seed(seed); m=MemoryModel(kind); opt=torch.optim.Adam(m.parameters(),lr=.003); rng=torch.Generator().manual_seed(seed+100); curve=[]
            for step in range(250):
                x,y=batch(64,32,rng); opt.zero_grad(); loss=nn.functional.cross_entropy(m(x),y); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); curve.append(loss.item())
            acc={}
            with torch.no_grad():
                for t in [8,32,64,128]:
                    x,y=batch(1024,t,torch.Generator().manual_seed(9000+t)); acc[str(t)]=(m(x).argmax(-1)==y).float().mean().item()
            r.append(dict(seed=seed,kind=kind,parameters=sum(p.numel() for p in m.parameters()),accuracy=acc,loss=curve)); print(kind,seed,acc,flush=True)
    for kind in ['RNN','GRU','LSTM','attention']:
        a=np.array([[v['accuracy'][str(t)] for t in [8,32,64,128]] for v in r if v['kind']==kind]); plt.errorbar([8,32,64,128],a.mean(0)*100,yerr=a.std(0)*100,marker='o',label=kind)
    plt.axhline(50,ls=':',label='Chance'); plt.xlabel('Sequence length'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'memory.png',dpi=180); plt.close()
    for kind in ['RNN','GRU','LSTM','attention']:
        a=np.array([v['loss'] for v in r if v['kind']==kind]); plt.plot(np.arange(250)+1,a.mean(0),label=kind)
    plt.xlabel('Training update'); plt.ylabel('Cross entropy'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'training.png',dpi=180); plt.close()
    (ROOT/'results.json').write_text(json.dumps(dict(bptt_errors=errors,runs=r),indent=2))
if __name__=='__main__': main()
