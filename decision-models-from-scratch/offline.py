"""Controlled synthetic permutation study; no pretrained weights needed."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
ROOT=Path(__file__).resolve().parent

def run(seed):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    x=rng.normal(size=(6000,4,8)).astype('float32')
    latent=x[:,:,0]+.6*x[:,:,1]+.3*x[:,:,2]*x[:,:,3]+rng.normal(0,.35,(6000,4))
    y=latent.argmax(1)
    # Canonical order is intentionally biased, even on the canonical test view.
    for i in range(len(y)):
        if rng.random()<.78:
            j=y[i]; x[i,[0,j]]=x[i,[j,0]]; y[i]=0
    x=torch.tensor(x); y=torch.tensor(y); results={}
    for name,pos,shuffle in [('biased',True,False),('shuffled',True,True),('shared',False,False)]:
        torch.manual_seed(seed); model=CandidateScorer(positional=pos); opt=torch.optim.Adam(model.parameters(),lr=.01)
        for epoch in range(20):
            for ids in torch.randperm(4000).split(128):
                a=x[ids]; b=y[ids]
                if shuffle:
                    perm=torch.rand(len(ids),4).argsort(1); a=a.gather(1,perm[:,:,None].expand(-1,-1,8)); b=(perm==b[:,None]).long().argmax(1)
                z=model(a); loss=decision_loss(z,F.one_hot(b,4).float()); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cal=model(x[4000:5000]).numpy(); test=model(x[5000:]).numpy(); truth=y[5000:].numpy(); pred=[]
            for perm in itertools.permutations(range(4)):
                p=np.array(perm); pred.append(p[model(x[5000:][:,p]).argmax(1).numpy()])
        pred=np.array(pred); t=fit_temperature(cal,y[4000:5000].numpy()); cp=probabilities(cal,t); tp=probabilities(test,t); threshold=fit_threshold(cp,y[4000:5000].numpy());
        results[name]=dict(canonical=metrics(probabilities(test),truth),permutation_accuracy=float((pred==truth).mean()),flip_rate=float((pred!=pred[0]).any(0).mean()),temperature=t,calibrated=metrics(tp,truth),threshold=threshold,selective=selective(tp,truth,threshold))
    return results

def main():
    torch.set_num_threads(1); runs=[run(s) for s in [7,19,42]]; names=list(runs[0]); summary={n:{k:float(np.mean([r[n][k] for r in runs])) for k in ['permutation_accuracy','flip_rate']} for n in names}
    for n in names:
        summary[n]['canonical_accuracy']=float(np.mean([r[n]['canonical']['accuracy'] for r in runs])); summary[n]['coverage']=float(np.mean([r[n]['selective']['coverage'] for r in runs]))
    report={'seeds':[7,19,42],'runs':runs,'mean':summary}; (ROOT/'results/offline.json').write_text(json.dumps(report,indent=2)); print(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,[100*summary[n]['flip_rate'] for n in names]); ax.set_ylabel('Questions with any answer flip (%)'); ax.set_title('Synthetic candidate-order stress test (3 seeds)'); fig.tight_layout(); fig.savefig(ROOT/'figures/permutation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); idx=np.arange(3); ax.bar(idx-.18,[100*summary[n]['canonical_accuracy'] for n in names],.36,label='Canonical'); ax.bar(idx+.18,[100*summary[n]['permutation_accuracy'] for n in names],.36,label='All permutations'); ax.set_xticks(idx,names); ax.set_ylabel('Accuracy (%)'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/accuracy.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
