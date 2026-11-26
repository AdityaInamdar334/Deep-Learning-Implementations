import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
F=ROOT/'figures'; F.mkdir(exist_ok=True)
torch.set_num_threads(1)

def quadratic():
    H=np.diag([1.,100.]); paths={}
    for name,lr in [('GD',.018),('Momentum',.01),('Nesterov',.01),('AdaGrad',.3),('RMSProp',.08),('Adam',.08),('Newton',1.)]:
        w=np.array([4.,2.]); v=np.zeros(2); r=v.copy(); m=v.copy(); path=[w.copy()]
        for t in range(1,201):
            g=H@w
            if name=='GD': w-=lr*g
            elif name in ('Momentum','Nesterov'):
                if name=='Nesterov': g=H@(w+.9*v)
                v=.9*v-lr*g; w+=v
            elif name=='Newton': w-=np.linalg.solve(H,g)
            else:
                r=r+g*g if name=='AdaGrad' else (.99*r+.01*g*g if name=='RMSProp' else .999*r+.001*g*g)
                if name=='Adam':
                    m=.9*m+.1*g; w-=lr*(m/(1-.9**t))/(np.sqrt(r/(1-.999**t))+1e-8)
                else: w-=lr*g/(np.sqrt(r)+1e-8)
            path.append(w.copy())
        paths[name]=np.array(path)
    x,y=np.meshgrid(np.linspace(-5,5,300),np.linspace(-2.5,2.5,300)); plt.figure(figsize=(9,5)); plt.contour(x,y,.5*(x*x+100*y*y),levels=np.geomspace(.1,350,14),colors='0.8')
    for n in ['GD','Momentum','Nesterov','Adam','Newton']:
        p=paths[n]; plt.plot(*p.T,label=n,alpha=.85)
    plt.legend();plt.xlabel('Low curvature coordinate');plt.ylabel('High curvature coordinate');plt.tight_layout();plt.savefig(F/'paths.png',dpi=180);plt.close()
    plt.figure(figsize=(9,5))
    for n,p in paths.items(): plt.semilogy(np.maximum(.5*np.sum((p@H)*p,axis=1),1e-24),label=n)
    plt.xlabel('Updates');plt.ylabel('Objective (floor 1e-24)');plt.legend();plt.tight_layout();plt.savefig(F/'quadratic_loss.png',dpi=180);plt.close()
    return {n:float(.5*p[-1]@H@p[-1]) for n,p in paths.items()}

def initialization():
    result={}
    for name in ['small','xavier','he','large']:
        torch.manual_seed(42); x=torch.randn(128,64); layers=[]; stats=[]
        for i in range(20):
            l=nn.Linear(64,64,bias=False)
            with torch.no_grad():
                if name=='xavier': nn.init.xavier_normal_(l.weight)
                elif name=='he': nn.init.kaiming_normal_(l.weight,nonlinearity='relu')
                else: nn.init.normal_(l.weight,std=.02 if name=='small' else .25)
            layers.append(l); x=l(x).relu(); stats.append(float(x.detach().square().mean().sqrt()))
        x.mean().backward(); result[name]={'activation_rms':stats,'gradient_rms':[float(l.weight.grad.square().mean().sqrt()) for l in layers]}
    for key in ['activation_rms','gradient_rms']:
        plt.figure(figsize=(9,5))
        for n,r in result.items(): plt.semilogy(range(1,21),np.maximum(r[key],1e-35),label=n)
        plt.xlabel('Layer (input to output)');plt.ylabel(key);plt.legend();plt.tight_layout();plt.savefig(F/(key+'.png'),dpi=180);plt.close()
    return result

def benchmark():
    X,y=load_digits(return_X_y=True); X=torch.tensor(X/16,dtype=torch.float32);y=torch.tensor(y,dtype=torch.long)
    tr,te=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42);tr,va=train_test_split(tr,test_size=.2,stratify=y[tr],random_state=43)
    configs={'SGD':(torch.optim.SGD,{'lr':.1}),'Momentum':(torch.optim.SGD,{'lr':.03,'momentum':.9}),'Nesterov':(torch.optim.SGD,{'lr':.03,'momentum':.9,'nesterov':True}),'AdaGrad':(torch.optim.Adagrad,{'lr':.03}),'RMSProp':(torch.optim.RMSprop,{'lr':.001}),'Adam':(torch.optim.Adam,{'lr':.001}),'AdamW':(torch.optim.AdamW,{'lr':.001,'weight_decay':.01})}
    rows=[];ce=nn.CrossEntropyLoss()
    for seed in [11,22,33]:
        for name,(cls,kw) in configs.items():
            torch.manual_seed(seed); model=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,10));opt=cls(model.parameters(),**kw);rng=np.random.default_rng(seed);history=[];best=float('inf')
            for epoch in range(30):
                model.train()
                for ids in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
                    opt.zero_grad(); loss=ce(model(X[ids]),y[ids]);loss.backward();opt.step()
                model.eval()
                with torch.no_grad():
                    a=model(X[tr]);b=model(X[va]);vl=float(ce(b,y[va]));history.append([float(ce(a,y[tr])),vl,float((b.argmax(1)==y[va]).float().mean())])
                if vl<best: best=vl;state={k:v.clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
            model.load_state_dict(state)
            with torch.no_grad():
                z=model(X[te]);acc=float((z.argmax(1)==y[te]).float().mean());nll=float(ce(z,y[te]))
            rows.append(dict(seed=seed,method=name,history=history,best_epoch=best_epoch,test_acc=acc,test_nll=nll));print(name,seed,round(acc,4),flush=True)
    plt.figure(figsize=(9,5))
    for n in configs:
        h=np.array([r['history'] for r in rows if r['method']==n]);plt.plot(range(1,31),h[:,:,1].mean(0),label=n)
    plt.xlabel('Epoch');plt.ylabel('Mean validation cross-entropy');plt.legend();plt.tight_layout();plt.savefig(F/'validation.png',dpi=180);plt.close()
    summary={n:{k:float(np.mean([r[k] for r in rows if r['method']==n])) for k in ['test_acc','test_nll','best_epoch']} for n in configs}
    return {'splits':[len(tr),len(va),len(te)],'runs':rows,'summary':summary}
if __name__=='__main__':
    result={'quadratic':quadratic(),'initialization':initialization(),'benchmark':benchmark()};(ROOT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['benchmark']['summary'],indent=2))
