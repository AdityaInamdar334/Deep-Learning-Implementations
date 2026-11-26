"""CPU demonstrations; not reproductions of the cited papers."""
import numpy as np
import torch
from torch import nn
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(1)
OUT=Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)

def tent(x):
    return 2*np.maximum(x,0)-4*np.maximum(x-.5,0)+2*np.maximum(x-1,0)

def depth_demo():
    x=np.linspace(0,1,4097); y=x.copy()
    fig,axes=plt.subplots(4,1,figsize=(8,7),sharex=True)
    for k,ax in enumerate(axes,1):
        y=tent(y); ax.plot(x,y); ax.set_ylabel(f'Depth {k}'); ax.set_ylim(-.05,1.05)
        print(f'Tent composition {k}: {2**k} affine pieces (analytic)')
    axes[-1].set_xlabel('Input'); fig.tight_layout(); fig.savefig(OUT/'depth.png',dpi=160); plt.close(fig)

def gradient_demo():
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,act,init in [('Sigmoid / Xavier',nn.Sigmoid, 'xavier'),('Tanh / Xavier',nn.Tanh,'xavier'),('ReLU / Xavier',nn.ReLU,'xavier'),('ReLU / He',nn.ReLU,'he')]:
        all_norms=[]
        for seed in range(10):
            torch.manual_seed(seed); x=torch.randn(64,64,requires_grad=True); h=x; hs=[x]
            for _ in range(24):
                layer=nn.Linear(64,64,bias=False)
                if init=='he': nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
                else: nn.init.xavier_normal_(layer.weight)
                h=act()(layer(h)); hs.append(h)
            # identical unit-norm output-gradient direction distribution per seed
            v=torch.randn_like(h); v=v/v.norm()
            gs=torch.autograd.grad((h*v).sum(),hs)
            all_norms.append([g.norm().item() for g in gs])
        med=np.median(all_norms,axis=0); ax.semilogy(range(25),med,label=name)
        print(f'{name}: median input gradient norm={med[0]:.6g}; output={med[-1]:.3f}')
    ax.set(xlabel='Layer index (0 = input)',ylabel='Median gradient L2 norm',title='Gradient propagation at initialization (10 seeds)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'gradients.png',dpi=160); plt.close(fig)

def data(seed,n):
    rng=np.random.default_rng(seed); x=rng.uniform(-1,1,(n,2)); y=(x[:,0]*x[:,1]<0).astype(np.float32)
    # independent observation noise, no label flips
    x=x+rng.normal(0,.08,x.shape)
    return torch.tensor(x,dtype=torch.float32),torch.tensor(y[:,None])

def generalization_demo():
    x,y=data(100,2048); xv,yv=data(101,1024); xt,yt=data(102,4096)
    fig,axes=plt.subplots(1,2,figsize=(9,4)); a=np.linspace(-1.25,1.25,180); xx,yy=np.meshgrid(a,a); grid=torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32)
    for nonlinear,ax in zip((False,True),axes):
        scores=[]
        for seed in range(5):
            torch.manual_seed(seed); m=nn.Sequential(nn.Linear(2,32),nn.ReLU() if nonlinear else nn.Identity(),nn.Linear(32,1)); opt=torch.optim.Adam(m.parameters(),lr=.02)
            best=float('inf'); state=None
            for epoch in range(200):
                for ids in torch.randperm(len(x)).split(128):
                    opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(m(x[ids]),y[ids]); loss.backward(); opt.step()
                with torch.no_grad(): vl=nn.functional.binary_cross_entropy_with_logits(m(xv),yv).item()
                if vl<best: best=vl; state={k:v.detach().clone() for k,v in m.state_dict().items()}
            m.load_state_dict(state)
            with torch.no_grad():
                acc=((m(xt)>0)==yt.bool()).float().mean().item(); scores.append(acc)
                if seed==0:
                    p=m(grid).sigmoid().numpy().reshape(xx.shape); ax.contourf(xx,yy,p,levels=np.linspace(0,1,11),cmap='coolwarm',alpha=.7); ax.scatter(xt[:180,0],xt[:180,1],c=yt[:180,0],cmap='coolwarm',s=7,edgecolors='k',linewidths=.2)
        name='ReLU' if nonlinear else 'Affine'; ax.set(title=name,xlabel='Observed x1',ylabel='Observed x2'); print(f'Noisy XOR {name}: test accuracy mean={np.mean(scores):.4f}, sample SD={np.std(scores,ddof=1):.4f}, seeds={scores}')
    fig.tight_layout(); fig.savefig(OUT/'noisy_xor.png',dpi=160); plt.close(fig)

if __name__=='__main__':
    depth_demo(); gradient_demo(); generalization_demo()
