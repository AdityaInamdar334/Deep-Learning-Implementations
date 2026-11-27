import json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
torch.set_num_threads(1)

def model(width): return nn.Sequential(nn.Linear(64,width),nn.ReLU(),nn.Linear(width,10))
def train(m,x,y,teacher=None):
    opt=torch.optim.AdamW(m.parameters(),lr=.003)
    for epoch in range(60):
        for ids in torch.randperm(len(x)).split(64):
            z=m(x[ids]); loss=F.cross_entropy(z,y[ids])
            if teacher is not None:
                with torch.no_grad(): p=(teacher(x[ids])/3).softmax(-1)
                loss=.4*loss+.6*9*F.kl_div((z/3).log_softmax(-1),p,reduction='batchmean')
            opt.zero_grad();loss.backward();opt.step()
    return m.eval()
def acc(m,x,y): return float((m(x).argmax(-1)==y).float().mean())
def quant(m,bits):
    import copy
    q=copy.deepcopy(m); scales=0; weights=0; biases=0
    with torch.no_grad():
        for l in q:
            if isinstance(l,nn.Linear):
                s=l.weight.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/(2**(bits-1)-1)
                l.weight.copy_((l.weight/s).round().clamp(-(2**(bits-1)-1),2**(bits-1)-1)*s)
                weights+=l.weight.numel();scales+=s.numel();biases+=l.bias.numel()
    return q,(weights*bits+32*(scales+biases)+7)//8

def plot(name, labels, vals, ylabel):
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(labels,vals);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIG/name,dpi=180);plt.close(fig)

def main():
    d=load_digits(); x=torch.tensor(d.data/16,dtype=torch.float32);y=torch.tensor(d.target)
    tr,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=42)
    va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=42)
    results={'environment':{'python':platform.python_version(),'torch':torch.__version__},'split':[len(tr),len(va),len(te)],'runs':[]}
    for seed in [11,22,33]:
        torch.manual_seed(seed); teacher=train(model(256),x[tr],y[tr]);torch.manual_seed(seed+100);student=train(model(16),x[tr],y[tr]);torch.manual_seed(seed+100); distilled=train(model(16),x[tr],y[tr],teacher)
        row={'seed':seed,'models':{}}
        for name,m in [('teacher',teacher),('student',student),('distilled',distilled)]:
            row['models'][name]={'accuracy':acc(m,x[te],y[te]),'parameters':sum(p.numel() for p in m.parameters())}
        row['quantization']={}
        for bits in [8,4,2]:
            q,storage=quant(teacher,bits);row['quantization'][str(bits)]={'accuracy':acc(q,x[te],y[te]),'ideal_payload_bytes':storage}
        with torch.no_grad():
            v=student(x[va]).softmax(-1); t=teacher(x[va]); thresholds=np.linspace(0,1,101)
            def eval_cascade(a,b,labels,threshold):
                route=a.max(-1).values<threshold; pred=torch.where(route,b.argmax(-1),a.argmax(-1));return float((pred==labels).float().mean()),float(route.float().mean())
            candidates=[(float(h),*eval_cascade(v,t,y[va],h)) for h in thresholds]
            feasible=[r for r in candidates if r[1]>=acc(teacher,x[va],y[va])-.01]
            chosen=min(feasible,key=lambda r:r[2]) if feasible else candidates[-1]
            ca,route=eval_cascade(student(x[te]).softmax(-1),teacher(x[te]),y[te],chosen[0]);row['cascade']={'threshold':chosen[0],'test_accuracy':ca,'teacher_fraction':route}
        results['runs'].append(row)
    mean=lambda f:float(np.mean([f(r) for r in results['runs']]))
    results['summary']={n:mean(lambda r:r['models'][n]['accuracy']) for n in ['teacher','student','distilled']}
    results['summary']['cascade_accuracy']=mean(lambda r:r['cascade']['test_accuracy']);results['summary']['teacher_fraction']=mean(lambda r:r['cascade']['teacher_fraction'])
    plot('distillation.png',list(results['summary'])[:3],[100*results['summary'][n] for n in ['teacher','student','distilled']],'Test accuracy (%)')
    plot('quantization.png',['FP32','INT8','INT4','INT2'],[100*results['summary']['teacher']]+[100*mean(lambda r:r['quantization'][str(b)]['accuracy']) for b in [8,4,2]],'Test accuracy (%)')
    plot('cascade.png',['Small only','Large only','Cascade'],[100*results['summary'][k] for k in ['student','teacher','cascade_accuracy']],'Test accuracy (%)')
    # CTC: enumerate all alignments for three frames and compare PyTorch.
    import itertools
    probs=torch.tensor([[.2,.7,.1],[.4,.5,.1],[.2,.2,.6]],dtype=torch.float64)
    def collapse(path):return [a for i,a in enumerate(path) if a!=0 and (i==0 or a!=path[i-1])]
    paths=[p for p in itertools.product(range(3),repeat=3) if collapse(p)==[1,2]]
    total=sum(float(np.prod([probs[t,a].item() for t,a in enumerate(p)])) for p in paths)
    loss=F.ctc_loss(probs.log().unsqueeze(1),torch.tensor([1,2]),torch.tensor([3]),torch.tensor([2]),reduction='sum')
    assert abs(float((-loss).exp())-total)<1e-12
    results['ctc']={'paths':len(paths),'probability':total,'loss':float(loss)}
    # Language modeling: same vocabulary and fixed synthetic held-out context.
    vocab=['a','b','c','d']; counts=Counter([('a','b'),('a','b'),('b','c'),('c','a')]); contexts=Counter(a for a,b in counts.elements()); p=[(counts['a',b]+1)/(contexts['a']+len(vocab)) for b in vocab];assert abs(sum(p)-1)<1e-12;results['ngram']={'vocabulary':vocab,'smoothed_probabilities':p}
    # Matrix factorization on synthetic low-rank ratings, split observed entries.
    rng=np.random.default_rng(42); R=rng.normal(size=(100,4))@rng.normal(size=(4,80));ids=rng.permutation(R.size);train_ids,test_ids=ids[:6400],ids[6400:];u,i=np.unravel_index(train_ids,R.shape);ut,it=np.unravel_index(test_ids,R.shape)
    torch.manual_seed(42);A=nn.Parameter(torch.randn(100,4)*.1);B=nn.Parameter(torch.randn(80,4)*.1);opt=torch.optim.Adam([A,B],lr=.05);target=torch.tensor(R[u,i],dtype=torch.float32)
    for _ in range(300):
        loss=((A[u]*B[i]).sum(-1)-target).square().mean();opt.zero_grad();loss.backward();opt.step()
    pred=(A[ut]*B[it]).sum(-1).detach().numpy();results['recommendation']={'mean_baseline_rmse':float(np.sqrt(np.mean((R[ut,it]-R[u,i].mean())**2))),'factor_rmse':float(np.sqrt(np.mean((R[ut,it]-pred)**2)))}
    (ROOT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__': main()
