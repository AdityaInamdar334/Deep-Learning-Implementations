import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn
ROOT=Path(__file__).parent; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
results={}; torch.set_num_threads(1)
def save(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=170); plt.close()
# A: repeated entities, random labels by entity
rng=np.random.default_rng(42); n=400; groups=np.repeat(np.arange(n),8); y=np.repeat(rng.integers(0,2,n),8)
X=np.repeat(rng.normal(size=(n,16)),8,axis=0)+rng.normal(0,.03,(n*8,16))
a,b=train_test_split(np.arange(len(y)),test_size=.25,random_state=42,stratify=y)
c,d=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(X,y,groups))
for name,tr,te in [('row',a,b),('group',c,d)]:
    m=KNeighborsClassifier(1).fit(X[tr],y[tr]); results[name]={'accuracy':float(m.score(X[te],y[te])),'overlap':len(set(groups[tr])&set(groups[te]))}
plt.bar(['Random rows','Unseen groups'],[results[k]['accuracy'] for k in ['row','group']]); plt.ylim(0,1.05); plt.ylabel('Accuracy'); save('leakage')
# B: rare event, threshold chosen on validation only
X,y=make_classification(n_samples=12000,n_features=12,n_informative=6,weights=[.97,.03],class_sep=1.2,random_state=42)
tr,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=42); va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=43)
m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X[tr],y[tr]); pv=m.predict_proba(X[va])[:,1]; pt=m.predict_proba(X[te])[:,1]
def cost(y,p,t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel(); return dict(cost=int(fp+20*fn),fp=int(fp),fn=int(fn),tp=int(tp),tn=int(tn),accuracy=float((tp+tn)/len(y)))
t=float(min(np.linspace(0,1,201),key=lambda t:cost(y[va],pv,t)['cost']))
results['threshold']={'chosen':t,'default':cost(y[te],pt,.5),'selected':cost(y[te],pt,t),'all_negative':cost(y[te],pt,2),'ap':float(average_precision_score(y[te],pt)),'prevalence':float(y[te].mean()),'brier':float(brier_score_loss(y[te],pt))}
p,r,_=precision_recall_curve(y[te],pt); plt.plot(r,p); plt.axhline(y[te].mean(),ls='--'); plt.xlabel('Recall'); plt.ylabel('Precision'); save('precision_recall')
# C: selective prediction on same independent test; fixed threshold policy sweep descriptive
X,y=load_digits(return_X_y=True); X=X/16; tr,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=42); va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=43)
m=LogisticRegression(max_iter=1000).fit(X[tr],y[tr]); pv=m.predict_proba(X[va]); pt=m.predict_proba(X[te]); thresholds=np.linspace(0,1,101)
def selective(p,y,t):
    mask=p.max(1)>=t; return float(mask.mean()),float((p.argmax(1)[mask]==y[mask]).mean()) if mask.any() else None
eligible=[t for t in thresholds if (lambda z:z[1] is not None and z[1]>=.98)(selective(pv,y[va],t))]; t=float(min(eligible)) if eligible else 1.
results['selective']={'threshold':t,'all':selective(pt,y[te],0),'chosen':selective(pt,y[te],t),'validation':selective(pv,y[va],t)}
curve=[selective(pt,y[te],t) for t in thresholds]; plt.plot([a for a,b in curve if b is not None],[b for a,b in curve if b is not None]); plt.xlabel('Coverage'); plt.ylabel('Accuracy among accepted'); save('coverage')
# D: learning curves matched nested data, three draws
rows=[]
for size in [40,80,160,320,640,1000]:
    for seed in [11,22,33]:
        ids=np.random.default_rng(seed).permutation(tr)[:size]
        for depth in [2,None]:
            m=DecisionTreeClassifier(max_depth=depth,random_state=seed).fit(X[ids],y[ids]); rows.append(dict(n=size,seed=seed,depth=str(depth),train=float(m.score(X[ids],y[ids])),val=float(m.score(X[va],y[va]))))
results['learning']=rows
for depth in ['2','None']:
    for split in ['train','val']:
        sizes=[40,80,160,320,640,1000]; plt.plot(sizes,[np.mean([r[split] for r in rows if r['n']==s and r['depth']==depth]) for s in sizes],marker='o',label=f'depth {depth}, {split}')
plt.xscale('log'); plt.xlabel('Training examples'); plt.ylabel('Accuracy'); plt.legend(); save('learning_curves')
# E: grid vs random, synthetic objective with one important coordinate
rng=np.random.default_rng(42); grid=np.array([(x,z) for x in np.linspace(0,1,6) for z in np.linspace(0,1,6)])
def objective(p): return (p[:,0]-.37)**2+.0001*(p[:,1]-.71)**2
G=float(objective(grid).min()); R=[float(objective(rng.random((36,2))).min()) for _ in range(200)]; results['search']={'grid':G,'random_median':float(np.median(R)),'random_win_fraction':float(np.mean(np.array(R)<G))}
plt.hist(R,bins=30); plt.axvline(G,label='Grid',ls='--'); plt.xlabel('Best objective after 36 evaluations'); plt.legend(); save('search')
# F: tiny-set sanity check and finite differences
x=torch.tensor(X[tr[:16]],dtype=torch.float32); target=torch.tensor(y[tr[:16]]); torch.manual_seed(42); net=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10)); opt=torch.optim.Adam(net.parameters(),lr=.01)
for _ in range(300):
    opt.zero_grad(); loss=nn.functional.cross_entropy(net(x),target); loss.backward(); opt.step()
acc=float((net(x).argmax(1)==target).float().mean()); w=np.array([.2,-.4,1.]); f=lambda w:np.sum(np.sin(w)+w*w); eps=1e-5; num=np.array([(f(w+eps*e)-f(w-eps*e))/(2*eps) for e in np.eye(3)]); exact=np.cos(w)+2*w
results['debug']={'tiny_accuracy':acc,'tiny_loss':float(loss.detach()),'gradient_max_error':float(np.max(abs(num-exact)))}; assert acc==1.; assert np.max(abs(num-exact))<1e-8
(ROOT/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps({k:v for k,v in results.items() if k!='learning'},indent=2))
