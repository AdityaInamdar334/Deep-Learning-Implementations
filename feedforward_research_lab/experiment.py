import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64)
y=np.array([[0],[1],[1],[0]],dtype=np.float64)

def forward(X,p):
    W,b,V,c=p
    z=X@W+b
    h=np.maximum(z,0)
    return h@V+c,z,h

def loss_grad(X,y,p):
    W,b,V,c=p
    pred,z,h=forward(X,p)
    r=pred-y
    d=2*r/len(X)
    dz=(d@V.T)*(z>0)
    return np.mean(r*r),[X.T@dz,dz.sum(0),h.T@d,d.sum(0)]

def main():
    A=np.c_[X,np.ones(len(X))]
    coef=np.linalg.lstsq(A,y,rcond=None)[0]
    print('Linear MSE:',np.mean((A@coef-y)**2))
    exact=[np.ones((2,2)),np.array([0.,-1.]),np.array([[1.],[-2.]]),np.zeros(1)]
    print('Exact XOR:',forward(X,exact)[0].ravel())
    rng=np.random.default_rng(7)
    p=[rng.normal(0,.5,(2,8)),np.full(8,.1),rng.normal(0,.3,(8,1)),np.zeros(1)]
    _,g=loss_grad(X,y,p)
    err=0.
    for a,ga in zip(p,g):
        for idx in np.ndindex(a.shape):
            old=a[idx]; eps=1e-6
            a[idx]=old+eps; plus=loss_grad(X,y,p)[0]
            a[idx]=old-eps; minus=loss_grad(X,y,p)[0]
            a[idx]=old
            err=max(err,abs((plus-minus)/(2*eps)-ga[idx]))
    print('Max finite difference error:',err)
    assert err<1e-6
    for _ in range(3000):
        _,g=loss_grad(X,y,p)
        for a,ga in zip(p,g): a-=.05*ga
    print('Trained NumPy MSE:',loss_grad(X,y,p)[0])
    print('Trained NumPy predictions:',forward(X,p)[0].ravel())
    xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    for nonlinear in (False,True):
        scores=[]
        for seed in range(5):
            torch.manual_seed(seed)
            model=nn.Sequential(nn.Linear(2,8),nn.ReLU() if nonlinear else nn.Identity(),nn.Linear(8,1))
            opt=torch.optim.Adam(model.parameters(),lr=.03)
            for _ in range(600):
                opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(xt),yt); loss.backward(); opt.step()
            with torch.no_grad(): scores.append(float(((model(xt)>0)==yt.bool()).float().mean()))
        print('ReLU' if nonlinear else 'Affine stack','training accuracies:',scores)
    z=torch.tensor(-10.,requires_grad=True)
    bce=nn.functional.binary_cross_entropy_with_logits(z,torch.ones_like(z))
    mse=.5*(z.sigmoid()-1)**2
    print('Wrong logit gradients: BCE',torch.autograd.grad(bce,z)[0].item(),'MSE',torch.autograd.grad(mse,z)[0].item())
if __name__=='__main__': main()
