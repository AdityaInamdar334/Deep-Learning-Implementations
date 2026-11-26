from pathlib import Path
import json, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
torch.set_num_threads(1)

def corr2d(x,k,stride=1,padding=0,dilation=1):
    x=np.pad(x,padding); kh,kw=k.shape
    eh,ew=dilation*(kh-1)+1,dilation*(kw-1)+1
    out=np.zeros(((x.shape[0]-eh)//stride+1,(x.shape[1]-ew)//stride+1))
    for i,j in np.ndindex(out.shape):
        out[i,j]=(x[i*stride:i*stride+eh:dilation,j*stride:j*stride+ew:dilation]*k).sum()
    return out

def lower_conv(x,w,b,stride=1,padding=0,dilation=1):
    patches=F.unfold(x,w.shape[-2:],dilation=dilation,padding=padding,stride=stride)
    y=w.flatten(1)@patches+b[None,:,None]
    h=(x.shape[-2]+2*padding-dilation*(w.shape[-2]-1)-1)//stride+1
    return y.reshape(x.shape[0],w.shape[0],h,-1)

def mechanics():
    torch.manual_seed(7)
    x=torch.randn(2,3,9,10,dtype=torch.float64,requires_grad=True)
    w=torch.randn(4,3,3,2,dtype=torch.float64,requires_grad=True)
    b=torch.randn(4,dtype=torch.float64,requires_grad=True)
    y=F.conv2d(x,w,b,stride=2,padding=1,dilation=2)
    z=lower_conv(x,w,b,stride=2,padding=1,dilation=2)
    torch.testing.assert_close(y,z)
    g=torch.randn_like(y)
    a=torch.autograd.grad(y,(x,w,b),g,retain_graph=True); c=torch.autograd.grad(z,(x,w,b),g)
    for u,v in zip(a,c): torch.testing.assert_close(u,v)
    np.testing.assert_allclose(corr2d(x.detach()[0,0].numpy(),w.detach()[0,0].numpy(),2,1,2),F.conv2d(x.detach()[0:1,0:1],w.detach()[0:1,0:1],stride=2,padding=1,dilation=2).numpy()[0,0],atol=1e-10)
    t=F.conv_transpose2d(g,w.detach(),stride=2,padding=1,dilation=2,output_padding=(0,1)); torch.testing.assert_close(t,a[0])
    def conv(v): return F.conv2d(F.pad(v,(1,1,1,1),mode='circular'),torch.ones(1,1,3,3,dtype=v.dtype))
    q=torch.randn(1,1,12,12,dtype=torch.float64)
    eq=(conv(q.roll(1,-1))-conv(q).roll(1,-1)).abs().max().item()
    img=np.zeros((12,12)); img[2:10,4:8]=1; k=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    fig,axs=plt.subplots(1,3,figsize=(9,3),layout='constrained')
    for ax,v,title in zip(axs,[img,k,corr2d(img,k,padding=1)],['Input','Vertical edge kernel','Signed response']):
        ax.imshow(v,cmap='gray' if title=='Input' else 'coolwarm'); ax.set_title(title); ax.axis('off')
    fig.savefig(FIG/'edge_detection.png',dpi=180); plt.close(fig)
    return {'forward_max_error':(y-z).abs().max().item(),'gradient_max_errors':[float((u-v).abs().max()) for u,v in zip(a,c)],'circular_shift_error':eq,'pool_same_grid_mean_change':(F.max_pool2d(q,2)-F.max_pool2d(q.roll(1,-1),2)).abs().mean().item()}

def shift(x,dy,dx):
    z=torch.zeros_like(x); h,w=x.shape[-2:]
    z[...,max(dy,0):min(h,h+dy),max(dx,0):min(w,w+dx)]=x[...,max(-dy,0):min(h,h-dy),max(-dx,0):min(w,w-dx)]
    return z

class Net(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind=='mlp':
            self.features=nn.Identity(); self.head=nn.Sequential(nn.Flatten(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,10))
        else:
            second=nn.Conv2d(16,32,3,padding=1) if kind!='separable' else nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16),nn.Conv2d(16,32,1))
            self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),second,nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)))
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,10))
    def forward(self,x): return self.head(self.features(x))

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        z=model(x); return {'accuracy':(z.argmax(1)==y).float().mean().item(),'loss':F.cross_entropy(z,y).item()}

def study():
    d=load_digits(); x=torch.tensor(d.images[:,None]/16,dtype=torch.float32); y=torch.tensor(d.target)
    x=F.pad(x,(4,4,4,4)); idx=np.arange(len(y))
    tr,te=train_test_split(idx,test_size=.25,stratify=y.numpy(),random_state=42)
    tr,va=train_test_split(tr,test_size=.2,stratify=y[tr].numpy(),random_state=42)
    perm=torch.randperm(256,generator=torch.Generator().manual_seed(123))
    rows=[]; histories=[]
    configs=[('mlp','centered'),('cnn','centered'),('separable','centered'),('mlp','permuted'),('cnn','permuted'),('cnn','augmented')]
    for seed in [0,1,2]:
        for kind,condition in configs:
            torch.manual_seed(seed); model=Net(kind); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
            rng=torch.Generator().manual_seed(seed+100); aug=torch.Generator().manual_seed(seed+200)
            def transform(v): return v.flatten(1)[:,perm].reshape_as(v) if condition=='permuted' else v
            best=float('inf'); history=[]
            for epoch in range(20):
                model.train()
                for ix in torch.randperm(len(tr),generator=rng).split(64):
                    xx=x[tr[ix]]; yy=y[tr[ix]]
                    if condition=='augmented':
                        offsets=torch.randint(-3,4,(len(xx),2),generator=aug)
                        xx=torch.cat([shift(v[None],int(dy),int(dx)) for v,(dy,dx) in zip(xx,offsets)])
                    opt.zero_grad(); loss=F.cross_entropy(model(transform(xx)),yy); loss.backward(); opt.step()
                val=evaluate(model,transform(x[va]),y[va]); history.append(val)
                if val['loss']<best: best=val['loss']; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
            model.load_state_dict(state)
            row={'seed':seed,'model':kind,'condition':condition,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'clean':evaluate(model,transform(x[te]),y[te]),'shifted':evaluate(model,transform(shift(x[te],3,3)),y[te])}
            rows.append(row); histories.append(history); print(row,flush=True)
            if seed==0 and kind=='cnn' and condition=='centered':
                weights=model.features[0].weight.detach().numpy()[:,0]; fig,axs=plt.subplots(2,8,figsize=(10,3),layout='constrained'); vmax=abs(weights).max()
                for ax,v in zip(axs.flat,weights): ax.imshow(v,cmap='coolwarm',vmin=-vmax,vmax=vmax); ax.axis('off')
                fig.suptitle('Learned first-layer filters (seed 0)'); fig.savefig(FIG/'learned_filters.png',dpi=180); plt.close(fig)
    labels=[f'{a}/{b}' for a,b in configs]; summary=[]
    for i,label in enumerate(labels):
        rr=rows[i::6]; summary.append({'configuration':label,'parameters':rr[0]['parameters'],**{f'{m}_{s}':float(getattr(np,s)([r[m]['accuracy']*100 for r in rr])) for m in ['clean','shifted'] for s in ['mean','std']}})
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained'); pos=np.arange(6)
    for off,m in [(-.18,'clean'),(.18,'shifted')]: ax.bar(pos+off,[r[m+'_mean'] for r in summary],.36,yerr=[r[m+'_std'] for r in summary],label=m,capsize=3)
    ax.set_xticks(pos,labels,rotation=25,ha='right'); ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(0,105); ax.legend(); fig.savefig(FIG/'accuracy.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for i,label in enumerate(labels): ax.plot(range(1,21),np.mean([[v['loss'] for v in h] for h in histories[i::6]],axis=0),label=label)
    ax.set(xlabel='Epoch',ylabel='Validation cross-entropy'); ax.legend(fontsize=8); fig.savefig(FIG/'learning_curves.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(8,3),layout='constrained')
    for i in range(5):
        axs[0,i].imshow(x[te[i],0],cmap='gray',vmin=0,vmax=1); axs[1,i].imshow(shift(x[te[i:i+1]],3,3)[0,0],cmap='gray',vmin=0,vmax=1)
        for ax in axs[:,i]: ax.axis('off')
    axs[0,0].set_title('Centered'); axs[1,0].set_title('Shifted'); fig.savefig(FIG/'shifted_digits.png',dpi=180); plt.close(fig)
    return {'split':[len(tr),len(va),len(te)],'runs':rows,'histories':histories,'summary':summary}

if __name__=='__main__':
    result={'mechanics':mechanics(),'study':study(),'torch':torch.__version__}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result['study']['summary'],indent=2))
