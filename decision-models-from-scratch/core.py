"""Shared scoring, calibration and selective prediction utilities."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CandidateScorer(nn.Module):
    def __init__(self, dim=8, positional=False, options=4):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,1))
        self.bias = nn.Parameter(torch.zeros(options)) if positional else None
    def forward(self,x):
        z=self.scorer(x).squeeze(-1)
        return z if self.bias is None else z+self.bias

def decision_loss(logits, targets, brier_weight=.1):
    logp=F.log_softmax(logits,dim=-1)
    return -(targets*logp).sum(-1).mean()+brier_weight*(logp.exp()-targets).square().sum(-1).mean()

def probabilities(z,t=1.):
    z=np.asarray(z)/t; e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def metrics(p,y):
    p=np.asarray(p); y=np.asarray(y); c=p.max(-1); ok=p.argmax(-1)==y
    ece=0.
    for i in range(15):
        m=(c>=i/15)&(c< (i+1)/15 if i<14 else c<=1)
        if m.any(): ece+=m.mean()*abs(c[m].mean()-ok[m].mean())
    return dict(accuracy=float(ok.mean()),nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),brier=float(((p-np.eye(p.shape[1])[y])**2).sum(-1).mean()),ece=float(ece))

def fit_temperature(z,y):
    return float(min(np.geomspace(.25,4,81),key=lambda t:metrics(probabilities(z,t),y)['nll']))

def selective(p,y,t):
    m=p.max(-1)>=t; ok=p.argmax(-1)==y
    return dict(coverage=float(m.mean()),accepted=int(m.sum()),accepted_accuracy=float(ok[m].mean()) if m.any() else None)

def fit_threshold(p,y,target=.98,min_count=30):
    for t in np.unique(np.r_[0,p.max(-1)]):
        r=selective(p,y,t)
        if r['accepted']>=min_count and r['accepted_accuracy']>=target: return float(t)
    return 1.01
