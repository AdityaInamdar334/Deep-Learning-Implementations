import json, copy, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
torch.set_num_threads(1)
EPOCHS = 100
SEEDS = [11, 22, 33]
CONFIGS = {'baseline': {}, 'dropout': {'drop': .3}, 'weight_decay': {'wd': .1}, 'noise': {'noise': .15}, 'smoothing': {'smooth': .1}, 'mixup': {'mix': .4}, 'combined': {'drop': .2, 'wd': .03, 'noise': .1}}

def model(drop=0):
    return nn.Sequential(nn.Linear(64,256),nn.ReLU(),nn.Dropout(drop),nn.Linear(256,256),nn.ReLU(),nn.Dropout(drop),nn.Linear(256,10))

@torch.no_grad()
def evaluate(net,x,y):
    net.eval(); logits=net(x); probs=logits.softmax(1); conf,pred=probs.max(1); correct=pred.eq(y)
    ece=sum(float(mask.float().mean() * (conf[mask].mean()-correct[mask].float().mean()).abs()) for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]) if (mask:=(conf>lo)&(conf<=hi)).any())
    return {'loss':float(F.cross_entropy(logits,y)), 'acc':float(correct.float().mean()),'ece':ece}

def run(seed,cfg):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); order=torch.Generator().manual_seed(seed+100)
    X,y=load_digits(return_X_y=True); X=X.astype('float32')/16
    a,xt,b,yt=train_test_split(X,y,test_size=450,stratify=y,random_state=seed)
    x,xv,y,yv=train_test_split(a,b,train_size=200,stratify=b,random_state=seed+1)
    x,xv,xt=map(torch.tensor,(x,xv,xt)); y,yv,yt=map(torch.tensor,(y,yv,yt))
    net=model(cfg.get('drop',0)); weights=[p for p in net.parameters() if p.ndim>1]; biases=[p for p in net.parameters() if p.ndim==1]
    opt=torch.optim.AdamW([{'params':weights,'weight_decay':cfg.get('wd',0)},{'params':biases,'weight_decay':0}],lr=.003)
    hist=[]; best=None; best_loss=float('inf'); best_epoch=0
    for epoch in range(1,EPOCHS+1):
        net.train()
        for idx in torch.randperm(len(x),generator=order).split(32):
            xb,yb=x[idx],y[idx]; opt.zero_grad()
            if cfg.get('noise'): xb=(xb+cfg['noise']*torch.randn_like(xb)).clamp(0,1)
            if cfg.get('mix'):
                lam=rng.beta(cfg['mix'],cfg['mix']); perm=torch.randperm(len(xb)); logits=net(lam*xb+(1-lam)*xb[perm]); loss=lam*F.cross_entropy(logits,yb)+(1-lam)*F.cross_entropy(logits,yb[perm])
            else: loss=F.cross_entropy(net(xb),yb,label_smoothing=cfg.get('smooth',0))
            loss.backward(); opt.step()
        tr=evaluate(net,x,y); va=evaluate(net,xv,yv); hist.append({'epoch':epoch,'train':tr,'val':va})
        if va['loss']<best_loss: best_loss=va['loss']; best=copy.deepcopy(net.state_dict()); best_epoch=epoch
    final=evaluate(net,xt,yt); net.load_state_dict(best); selected=evaluate(net,xt,yt)
    return {'seed':seed,'best_epoch':best_epoch,'final':final,'selected':selected,'history':hist}

def main():
    results={name:[run(s,cfg) for s in SEEDS] for name,cfg in CONFIGS.items()}
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
    for name,runs in results.items():
        print(name, 'final',np.mean([r['final']['acc'] for r in runs]),'selected',np.mean([r['selected']['acc'] for r in runs]), 'nll',np.mean([r['selected']['loss'] for r in runs]),'ece',np.mean([r['selected']['ece'] for r in runs]),'epochs',[r['best_epoch'] for r in runs])
    for metric in ['loss','acc']:
        plt.figure(figsize=(9,5))
        for name in results:
            plt.plot(np.arange(1,EPOCHS+1),np.mean([[h['val'][metric] for h in r['history']] for r in results[name]],axis=0),label=name)
        plt.xlabel('Epoch');plt.ylabel('Validation '+metric);plt.legend(ncol=2);plt.tight_layout();plt.savefig(FIG/f'validation_{metric}.png',dpi=180);plt.close()
    plt.figure(figsize=(9,5)); runs=results['baseline'];
    for split in ['train','val']: plt.plot(np.arange(1,EPOCHS+1),np.mean([[h[split]['loss'] for h in r['history']] for r in runs],axis=0),label=split)
    plt.xlabel('Epoch');plt.ylabel('Cross entropy');plt.legend();plt.tight_layout();plt.savefig(FIG/'baseline_gap.png',dpi=180);plt.close()
    plt.figure(figsize=(10,5)); vals=np.array([[r['selected']['acc'] for r in results[n]] for n in CONFIGS]);plt.bar(list(CONFIGS),vals.mean(1),yerr=vals.std(1,ddof=1),capsize=4);plt.ylim(0,1);plt.ylabel('Held-out accuracy');plt.xticks(rotation=20);plt.tight_layout();plt.savefig(FIG/'ablation.png',dpi=180);plt.close()
    (ROOT/'environment.txt').write_text(f'Python {platform.python_version()}\nTorch {torch.__version__}\nNumPy {np.__version__}\nScikit-learn {sklearn.__version__}\nMatplotlib {matplotlib.__version__}\n')
if __name__=='__main__': main()
