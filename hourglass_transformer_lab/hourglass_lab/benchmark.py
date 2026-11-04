"""Multi-seed educational benchmark. Do not treat as published Hourglass replication."""
from __future__ import annotations
import argparse, json, time, statistics, platform, random
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import HourglassLM,VanillaLM
from .data import repeat_batch


def configurations():
    return {
      'vanilla_4': dict(kind='vanilla'),
      'hourglass_k2': dict(kind='hourglass',factors=(2,)),
      'hourglass_k3': dict(kind='hourglass',factors=(3,)),
      'hourglass_k4': dict(kind='hourglass',factors=(4,)),
      'hourglass_linear_k3': dict(kind='hourglass',factors=(3,),pooling='linear'),
      'hourglass_unshifted_UNSAFE': dict(kind='hourglass',factors=(3,),do_shift=False),
      'hourglass_no_skip': dict(kind='hourglass',factors=(3,),skip=False),
      'hourglass_factor_dropout': dict(kind='hourglass',factors=(3,),vary_factor=True),
    }


def model_for(cfg,width,vocab):
    args=cfg.copy();kind=args.pop('kind');args.pop('vary_factor',None)
    return VanillaLM(vocab=vocab,width=width,heads=4,layers=4) if kind=='vanilla' else HourglassLM(vocab=vocab,width=width,heads=4,**args)

@torch.inference_mode()
def score(model, seed, length, period, vocab, examples=192):
    model.eval()
    s=repeat_batch(examples,length,period=period,vocab=vocab,seed=seed)
    logits=model(s[:,:-1])
    # Score all next tokens and the positions with a predictable copy rule.
    loss=F.cross_entropy(logits.reshape(-1,vocab),s[:,1:].reshape(-1)).item()
    start=max(0,period-1)
    pred=logits[:,start:].argmax(-1)
    targ=s[:,1+start:]
    acc=(pred==targ).float().mean().item()
    return {'cross_entropy_nats':round(loss,6),'copy_region_accuracy':round(acc,6),
            'perplexity':round(float(torch.exp(torch.tensor(loss))),5)}


def model_causal_difference(model,length=25,vocab=32):
    model.eval()
    with torch.no_grad():
        inp=repeat_batch(1,length,vocab=vocab,seed=321)[:,:-1]
        altered=inp.clone();altered[:,5]=(altered[:,5]+13)%vocab
        a=model(inp);b=model(altered)
        return float((a[:,:5]-b[:,:5]).abs().max().item())


def single_run(name,cfg,seed,steps,length,width,vocab,batch,lr,eval_size=192):
    torch.manual_seed(seed)
    random.seed(seed)
    model=model_for(cfg,width,vocab)
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    factor_rng=random.Random(seed+8129)
    model.train()
    losses=[]
    t0=time.perf_counter()
    for step in range(steps):
        s=repeat_batch(batch,length,period=4,vocab=vocab,seed=seed*100000+step+11)
        optimizer.zero_grad(set_to_none=True)
        factor=factor_rng.choice((2,3)) if cfg.get('vary_factor') else None
        logits=model(s[:,:-1],override_factor=factor) if factor else model(s[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,vocab),s[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(loss.item())
    train_seconds=time.perf_counter()-t0
    in_dist=score(model,seed=50000+seed,length=length,period=4,vocab=vocab,examples=eval_size)
    shift=score(model,seed=60000+seed,length=length,period=7,vocab=vocab,examples=eval_size)
    longer=score(model,seed=70000+seed,length=length+16,period=4,vocab=vocab,examples=eval_size)
    causal_difference=model_causal_difference(model,length=min(25,length),vocab=vocab)
    return {'name':name,'seed':seed,'parameters':parameters,'steps':steps,
            'train_seconds':round(train_seconds,4),'mean_last_10_train_loss':round(statistics.mean(losses[-10:]),5),
            'in_distribution':in_dist,'period_shift':shift,'longer_context':longer,
            'causal_leak_max_abs':round(causal_difference,7)}


def aggregate(trials):
    output={}
    for name in sorted(set(t['name'] for t in trials)):
        rs=[t for t in trials if t['name']==name]
        keys={
            'params':[t['parameters'] for t in rs],
            'train_seconds':[t['train_seconds'] for t in rs],
            'familiar_accuracy':[100*t['in_distribution']['copy_region_accuracy'] for t in rs],
            'shifted_accuracy':[100*t['period_shift']['copy_region_accuracy'] for t in rs],
            'longer_accuracy':[100*t['longer_context']['copy_region_accuracy'] for t in rs],
            'familiar_ppl':[t['in_distribution']['perplexity'] for t in rs],
            'causal_diff':[t['causal_leak_max_abs'] for t in rs],
        }
        output[name]={k:{'mean':round(statistics.mean(v),5),'std':round(statistics.stdev(v),5) if len(v)>1 else 0} for k,v in keys.items()}
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--steps',type=int,default=90)
    ap.add_argument('--length',type=int,default=48)
    ap.add_argument('--width',type=int,default=32)
    ap.add_argument('--vocab',type=int,default=32)
    ap.add_argument('--batch-size',type=int,default=20)
    ap.add_argument('--lr',type=float,default=0.002)
    ap.add_argument('--eval-size',type=int,default=192)
    ap.add_argument('--output',default='results/benchmark.json')
    ap.add_argument('--only',nargs='*',default=None,help='Optional configuration names for a reduced run')
    args=ap.parse_args()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    trials=[]
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for name,cfg in configurations().items():
        if args.only is not None and name not in args.only: continue
        for seed in range(args.seeds):
            r=single_run(name,cfg,seed,args.steps,args.length,args.width,args.vocab,args.batch_size,args.lr,args.eval_size)
            trials.append(r)
            # Save after each completed trial; a slow run never loses all progress.
            path.write_text(json.dumps({'methodology':'Synthetic repeat task. Fixed steps, not equal parameters or FLOPs. Unshifted is an intentionally invalid causal model.', 'settings':vars(args),'environment':{'python':platform.python_version(),'torch':torch.__version__, 'threads':torch.get_num_threads()}, 'trials':trials,'summary':aggregate(trials)},indent=2))
            print(f"{name:29s} seed={seed} familiar={r['in_distribution']['copy_region_accuracy']:.3f} shifted={r['period_shift']['copy_region_accuracy']:.3f} sec={r['train_seconds']:.2f} leak={r['causal_leak_max_abs']:.5f}",flush=True)
    result={'methodology':'Synthetic repeat task. Fixed steps, not equal parameters or FLOPs. Unshifted is an intentionally invalid causal model.',
            'settings':vars(args),'environment':{'python':platform.python_version(),'torch':torch.__version__, 'threads':torch.get_num_threads()},
            'trials':trials,'summary':aggregate(trials)}
    path.write_text(json.dumps(result,indent=2))
    print('Saved',path)
    print('Summary:')
    for name,vals in result['summary'].items():
        print(name, 'familiar',vals['familiar_accuracy']['mean'], 'shifted',vals['shifted_accuracy']['mean'], 'sec',vals['train_seconds']['mean'])

if __name__=='__main__': main()
