"""Reproducible CPU experiments for linear factor models. Run python lab.py."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, DictionaryLearning
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent

def ppca(x,k):
    mu=x.mean(0); xc=x-mu
    v,u=np.linalg.eigh(xc.T@xc/len(x)); v=v[::-1]; u=u[:,::-1]
    noise=max(v[k:].mean(),1e-10)
    return mu,u[:,:k]*np.sqrt(np.maximum(v[:k]-noise,0)),noise

def posterior(x,mu,w,noise):
    m=w.T@w+noise*np.eye(w.shape[1])
    return np.linalg.solve(m,w.T@(x-mu).T).T,noise*np.linalg.inv(m)

def sfa(x,k):
    mu=x.mean(0); z=x-mu; d=np.diff(z,axis=0)
    v,u=eigh(d.T@d/len(d),z.T@z/len(z)+1e-9*np.eye(z.shape[1]))
    return mu,u[:,:k],v[:k]

def recovery(a,b):
    k=a.shape[1]; c=np.abs(np.corrcoef(a.T,b.T)[:k,k:]); i,j=linear_sum_assignment(-c)
    return float(c[i,j].mean())

def ista(x,d,lam,steps=250):
    h=np.zeros((len(x),d.shape[1])); eta=1/max(np.linalg.norm(d,2)**2,1e-12)
    for _ in range(steps):
        z=h-eta*((h@d.T-x)@d); h=np.sign(z)*np.maximum(np.abs(z)-eta*lam,0)
    return h

def savefig(name):
    plt.tight_layout(); plt.savefig(ROOT/'figures'/name,dpi=180); plt.close()

def main():
    torch.set_num_threads(1); r=np.random.default_rng(42); results={}
    # 1: independent nuisance and perfectly predictive low variance feature
    y=r.integers(0,2,6000); x=np.c_[r.normal(0,10,len(y)),2*y-1]; tr=x[:4000]; te=x[4000:]; p=PCA(1).fit(tr)
    a=LogisticRegression().fit(tr,y[:4000]).score(te,y[4000:]); b=LogisticRegression().fit(p.transform(tr),y[:4000]).score(p.transform(te),y[4000:]); results['pca']={'variance':float(p.explained_variance_ratio_[0]),'full_accuracy':a,'pca_accuracy':b}
    plt.bar(['Retained variance','Full input accuracy','PCA-only accuracy'],[p.explained_variance_ratio_[0],a,b]); plt.ylim(0,1.05); plt.ylabel('Fraction'); savefig('pca_failure.png')
    # 2: Gaussian factor model with heterogeneous sensor noise
    w=r.normal(size=(8,2)); noise=np.array([.03,.05,.08,.15,.4,1,2,4]); x=r.normal(size=(7000,2))@w.T+r.normal(size=(7000,8))*np.sqrt(noise); tr,te=x[:5000],x[5000:]
    mu,wp,s=ppca(tr,2); f=FactorAnalysis(2,random_state=42).fit(tr); ll=multivariate_normal.logpdf(te,mean=mu,cov=wp@wp.T+s*np.eye(8)).mean(); results['noise']={'ppca_loglik':float(ll),'fa_loglik':float(f.score(te)),'ppca_noise':float(s),'fa_noise':f.noise_variance_.tolist(),'true_noise':noise.tolist()}
    plt.plot(noise,'o-',label='True'); plt.plot(f.noise_variance_,'s-',label='FA'); plt.axhline(s,label='PPCA'); plt.xlabel('Sensor'); plt.ylabel('Noise variance'); plt.legend(); savefig('noise.png')
    # 3: ICA on independent non-Gaussian sources, held-out recovery
    h=np.c_[r.laplace(size=6000),r.uniform(-2,2,6000)]; h=(h-h[:4000].mean(0))/h[:4000].std(0); x=h@np.array([[1,.8],[.4,1.2]]).T; p=PCA(2).fit(x[:4000]); i=FastICA(2,whiten='unit-variance',random_state=42,max_iter=2000).fit(x[:4000]); hp=p.transform(x[4000:]); hi=i.transform(x[4000:]); results['ica']={'pca_recovery':recovery(h[4000:],hp),'ica_recovery':recovery(h[4000:],hi)}
    plt.bar(['PCA','FastICA'],list(results['ica'].values())); plt.ylim(0,1.05); plt.ylabel('Matched absolute correlation'); savefig('ica.png')
    # 4: time structure, train/test contiguous segments
    t=np.arange(6000); h=np.c_[np.sin(t*.013),np.sin(t*.7)]; x=h@np.array([[.3,2],[.2,-1]]).T+.02*r.normal(size=(len(t),2)); mu,w,v=sfa(x[:4000],1); p=PCA(1).fit(x[:4000]); ys=((x[4000:]-mu)@w)[:,0]; yp=p.transform(x[4000:])[:,0]; slow=h[4000:,0]; results['sfa']={'pca_slow_correlation':float(abs(np.corrcoef(yp,slow)[0,1])),'sfa_slow_correlation':float(abs(np.corrcoef(ys,slow)[0,1]))}
    plt.plot(slow[:500],label='True slow factor'); plt.plot(ys[:500]*np.sign(np.corrcoef(ys,slow)[0,1])/ys.std()*slow.std(),label='SFA (rescaled)',alpha=.8); plt.legend(); plt.xlabel('Held-out timestep'); savefig('sfa.png')
    # 5: known dictionary, sparse inference tradeoff
    d=r.normal(size=(16,32)); d/=np.linalg.norm(d,axis=0); h=np.zeros((1200,32)); idx=np.array([r.choice(32,3,replace=False) for _ in h]); np.put_along_axis(h,idx,r.normal(size=idx.shape),axis=1); x=h@d.T+.01*r.normal(size=(len(h),16)); rows=[]
    for lam in [.01,.03,.1,.3,1.]:
        z=ista(x[800:],d,lam); rows.append({'lambda':lam,'mse':float(np.mean((z@d.T-x[800:])**2)),'zero_fraction':float(np.mean(abs(z)<1e-8))})
    results['ista']=rows; plt.plot([q['zero_fraction'] for q in rows],[q['mse'] for q in rows],'o-'); plt.xlabel('Fraction zero'); plt.ylabel('Reconstruction MSE'); savefig('sparse_tradeoff.png')
    # 6: learned dictionary vs known dictionary (independent held-out examples)
    dl=DictionaryLearning(n_components=32,alpha=.1,max_iter=30,fit_algorithm='cd',transform_algorithm='lasso_cd',transform_alpha=.1,random_state=42).fit(x[:800]); z=dl.transform(x[800:]); results['dictionary']={'mse':float(np.mean((z@dl.components_-x[800:])**2)),'zero_fraction':float(np.mean(abs(z)<1e-8))}
    # 7: SAE on synthetic nonnegative sparse activations, two objectives, three seeds
    rows=[]
    for seed in [11,22,33]:
        rng=np.random.default_rng(seed); torch.manual_seed(seed); d=rng.normal(size=(16,32)); d/=np.linalg.norm(d,axis=0); h=np.zeros((3500,32)); idx=np.array([rng.choice(32,3,replace=False) for _ in h]); np.put_along_axis(h,idx,rng.uniform(.3,2,size=idx.shape),axis=1); data=torch.tensor(h@d.T,dtype=torch.float32); tr,te=data[:3000],data[3000:]
        for lam in [0.,.03,.1]:
            torch.manual_seed(seed); enc=nn.Linear(16,32); dec=nn.Linear(32,16); opt=torch.optim.Adam([*enc.parameters(),*dec.parameters()],lr=.003); g=torch.Generator().manual_seed(seed)
            for epoch in range(60):
                for ids in torch.randperm(len(tr),generator=g).split(128):
                    z=enc(tr[ids]).relu(); loss=(dec(z)-tr[ids]).square().mean()+lam*z.mean(); opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad(): dec.weight.div_(dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
            with torch.no_grad():
                z=enc(te).relu(); rows.append({'seed':seed,'lambda':lam,'mse':float((dec(z)-te).square().mean()),'active':float((z>0).sum(1).float().mean())})
    results['sae']=rows; plt.plot([np.mean([v['active'] for v in rows if v['lambda']==l]) for l in [0.,.03,.1]],[np.mean([v['mse'] for v in rows if v['lambda']==l]) for l in [0.,.03,.1]],'o-'); plt.xlabel('Mean active features'); plt.ylabel('Held-out MSE'); savefig('sae.png')
    # 8: exact algebraic non-identifiability
    h=r.normal(size=(100,3)); w=r.normal(size=(6,3)); q=np.array([[1.,.5,0],[0,2,0],[0,0,.8]]); results['identifiability']={'max_reconstruction_difference':float(np.max(abs(h@w.T-(h@q.T)@(w@np.linalg.inv(q)).T)))}
    (ROOT/'results'/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
