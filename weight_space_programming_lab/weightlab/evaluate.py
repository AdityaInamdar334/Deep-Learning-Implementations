"""Independent holdout tasks and navigation rollouts used only after edits."""
from __future__ import annotations
import torch
from .core import TinyPolicy

@torch.no_grad()
def classification_accuracy(model: TinyPolicy, *, seed=10, n=3000, shift=1.0) -> dict:
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,4,generator=g)
    x[:,3]*=shift
    labels=torch.where(x[:,0]>0, 0, 1)
    pred=model(x).argmax(dim=-1)
    return {'accuracy':float((pred==labels).float().mean()),'stall_rate':float((pred==2).float().mean()),'n':n}

@torch.no_grad()
def navigation_success(model: TinyPolicy, *, seed=20, episodes=300, horizon=14, shift=1.0):
    """No reward learning, just measured task success on unseen random initial states."""
    g=torch.Generator().manual_seed(seed)
    x=(torch.rand(episodes,generator=g)*2-1)*1.8
    done=x.abs()<0.12
    steps=torch.zeros(episodes)
    for t in range(horizon):
        obs=torch.randn(episodes,4,generator=g)
        obs[:,0]=x
        obs[:,3]*=shift
        a=model(obs).argmax(-1)
        move=torch.where(a==0, -0.2,torch.where(a==1,0.2,0.0))
        x = x+torch.where(done,0.0,move)
        freshly=(~done)&(x.abs()<0.12)
        steps[freshly]=t+1
        done=done|freshly
    return {'success_rate':float(done.float().mean()), 'mean_steps_success':float(steps[done].mean()) if bool(done.any()) else 0., 'episodes': episodes}
