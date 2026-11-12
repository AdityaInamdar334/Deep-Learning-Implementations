"""Optional, expensive Fashion-MNIST-scale protocol approximation.

This follows the *publicly described experiment*, but has not been validated
against its author's unpublished source/archive. Run small before --full.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .experiment import label_for_epoch
from .topology import normalize_rows, summarize

PAPER_SEEDS = (42, 123, 456, 789, 2026)
CHECKPOINTS = tuple(range(0, 11)) + (15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
LAYERS = ('fc1', 'fc2', 'fc3', 'fc4', 'fc5')


def paper_style_model():
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            sizes = (784, 1024, 512, 256, 128, 64)
            self.layers = nn.ModuleList([nn.Linear(a,b) for a,b in zip(sizes,sizes[1:])])
            self.out = nn.Linear(64,10)
        def forward(self, x):
            import torch.nn.functional as F
            for layer in self.layers: x=F.relu(layer(x))
            return self.out(x)
    return MLP()


def fixed_row_indices(n, count):
    if n <= count: return np.arange(n)
    return np.sort(np.random.default_rng(2718+n).choice(n, size=count, replace=False))


def load_fashion(data_dir):
    import torch
    from torchvision.datasets import FashionMNIST
    train=FashionMNIST(data_dir, train=True, download=True)
    test=FashionMNIST(data_dir, train=False, download=True)
    arr=train.data.numpy(); y=train.targets.numpy()
    rng=np.random.default_rng(314159)
    fitids=[];valids=[]
    for klass in range(10):
        ids=np.flatnonzero(y==klass)
        rng.shuffle(ids)
        fitids.extend(ids[600:]); valids.extend(ids[:600])
    fitids=np.array(fitids);valids=np.array(valids)
    fit=arr[fitids].reshape(-1,784).astype('float32')/255.
    valid=arr[valids].reshape(-1,784).astype('float32')/255.
    testx=test.data.numpy().reshape(-1,784).astype('float32')/255.
    mean=float(fit.mean()); std=float(fit.std())
    fit=(fit-mean)/max(std,1e-12);valid=(valid-mean)/max(std,1e-12);testx=(testx-mean)/max(std,1e-12)
    return (torch.from_numpy(fit),torch.from_numpy(y[fitids].astype('int64')),
            torch.from_numpy(valid),torch.from_numpy(y[valids].astype('int64')),
            torch.from_numpy(testx),test.targets.long(),mean,std)


def cloud_metrics(model, epoch, condition, seed):
    from .ripser_optional import finite_intervals, rips_diagrams
    rows=[]
    for name,layer in zip(LAYERS,model.layers):
        w=layer.weight.detach().cpu().numpy()
        n=min(128,w.shape[0])
        indices=fixed_row_indices(w.shape[0],n)
        for rep in ('raw','normalized'):
            pts=w[indices]
            dgms=rips_diagrams(pts,normalize=(rep=='normalized'),maxdim=1)
            for dim in (0,1):
                s=summarize(finite_intervals(dgms[dim]))
                rows.append(dict(seed=seed, condition=condition,epoch=epoch,layer=name,
                                 representation=rep,homology=f'H{dim}',**s))
    return rows


def run_protocol(data_dir='data',output='results/fashion',epochs=2,limit=6000,
                 seeds=(42,),save_weights=False):
    """Full architecture, optional subset for quick start; no published-value claim."""
    import torch
    import torch.nn.functional as F
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    trainx,trainy,valx,valy,testx,testy,mean,std=load_fashion(data_dir)
    if limit:
        # exact matched subset across all three label conditions, not full paper
        rng=np.random.default_rng(4491)
        ids=np.sort(rng.choice(len(trainy),size=min(limit,len(trainy)),replace=False))
        trainx=trainx[ids]; trainy=trainy[ids]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    records=[];train_history=[]
    for seed in seeds:
        torch.manual_seed(seed)
        starting=paper_style_model()
        initial=copy.deepcopy(starting.state_dict())
        for condition in ('correct','fixed_random','changing_random'):
            model=paper_style_model().to(device)
            model.load_state_dict(initial)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for epoch in range(epochs+1):
                if epoch>0:
                    targets=torch.as_tensor(label_for_epoch(trainy.numpy(),condition,seed,epoch-1),device=device)
                    order=np.random.default_rng(np.random.SeedSequence([seed,29,epoch])).permutation(len(trainy))
                    model.train()
                    for start in range(0,len(trainy),256):
                        ids=order[start:start+256]
                        xx=trainx[ids].to(device)
                        yy=targets[ids]
                        opt.zero_grad(set_to_none=True)
                        loss=F.cross_entropy(model(xx),yy)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        opt.step()
                model.eval()
                with torch.no_grad():
                    # held-out accuracy only at checkpoint; test reserved for end
                    predictions=[]
                    for start in range(0,len(valx),256):
                        predictions.append(model(valx[start:start+256].to(device)).argmax(1).cpu())
                    valacc=float((torch.cat(predictions)==valy).float().mean())
                train_history.append(dict(seed=seed,condition=condition,epoch=epoch,
                                          true_validation_accuracy=valacc,
                                          full_paper_data=limit==0 and epochs==100 and len(seeds)==5))
                if epoch in CHECKPOINTS or epoch==epochs:
                    records.extend(cloud_metrics(model,epoch,condition,seed))
                    if save_weights:
                        path=out/f'model_seed{seed}_{condition}_epoch{epoch}.pt'
                        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
            with torch.no_grad():
                pred=[]
                for start in range(0,len(testx),256):
                    pred.append(model(testx[start:start+256].to(device)).argmax(1).cpu())
                testacc=float((torch.cat(pred)==testy).float().mean())
            print(f'seed={seed} condition={condition} test_true_accuracy={testacc:.4f}')
    for filename,data in [('topology.csv',records),('history.csv',train_history)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'manifest.json').write_text(json.dumps(dict(seeds=list(seeds),epochs=epochs,fit_rows=len(trainy),
                source="arxiv:2611.13701 method-inspired; NOT author-validated code",
                normalization_mean=mean,normalization_std=std,device=device),indent=2))
    return records


def main():
    p=argparse.ArgumentParser(description='Optional Fashion-MNIST-scale method demo; expensive')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output',default='results/fashion')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--limit',type=int,default=6000,help='0=all 54k fitting samples')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--full',action='store_true',help='100 epochs, all 5 seeds, all training data; expensive')
    p.add_argument('--save-weights',action='store_true')
    args=p.parse_args()
    if args.full:
        args.epochs=100;args.limit=0;args.seeds=list(PAPER_SEEDS)
    run_protocol(args.data_dir,args.output,args.epochs,args.limit,args.seeds,args.save_weights)

if __name__=='__main__': main()
