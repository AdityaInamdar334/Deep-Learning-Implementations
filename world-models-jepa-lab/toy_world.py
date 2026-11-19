"""Small observable world-model exercise. NOT an implementation of JEPA.
A moving object follows a learnable action-conditioned rule while background pixels
are resampled independently, making exact next-frame reconstruction impossible.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

SIZE=24
ACTIONS=np.array([[1,0],[-1,0],[0,1],[0,-1],[0,0]],dtype=float)

def render(position,rng,noise=0.25):
    frame=rng.uniform(0,noise,(SIZE,SIZE)).astype(np.float32)
    x,y=np.clip(np.rint(position).astype(int),0,SIZE-1)
    frame[y,x]=1.0
    return frame

def make_data(n=1200,seed=3,noise=0.25):
    rng=np.random.default_rng(seed)
    xs=[];acts=[];ys=[];before=[];after=[]
    for _ in range(n):
        pos=rng.integers(3,SIZE-3,size=2).astype(float)
        a=int(rng.integers(0,len(ACTIONS)))
        nxt=pos+ACTIONS[a]
        xs.append(pos); acts.append(a); ys.append(nxt)
        before.append(render(pos,rng,noise));after.append(render(nxt,rng,noise))
    return np.array(xs),np.array(acts),np.array(ys),np.array(before),np.array(after)

def fit_transition(states,actions,next_states):
    onehot=np.eye(len(ACTIONS))[actions]
    design=np.column_stack([states,onehot,np.ones(len(states))])
    return np.linalg.lstsq(design,next_states,rcond=None)[0]

def predict(states,actions,weights):
    return np.column_stack([states,np.eye(len(ACTIONS))[actions],np.ones(len(states))])@weights

def evaluate(train_n=1000,test_n=200,seed=3,noise=0.25):
    x,a,y,frames,targets=make_data(train_n,seed,noise)
    tx,ta,ty,tframes,ttargets=make_data(test_n,seed+101,noise)
    W=fit_transition(x,a,y)
    pred=predict(tx,ta,W)
    # Pixel persistence: a defensible no-training baseline, not an optimized video predictor.
    baseline_pixel_mse=float(np.mean((tframes-ttargets)**2))
    # Oracle-action move foreground while background predicted as its conditional expectation.
    oracle=np.full_like(ttargets,noise/2)
    for i,(px,py) in enumerate(pred):
        ix,iy=np.clip(np.rint([px,py]).astype(int),0,SIZE-1)
        oracle[i,iy,ix]=1.
    structured_pixel_mse=float(np.mean((oracle-ttargets)**2))
    state_mae=float(np.mean(np.abs(pred-ty)))
    state_round_accuracy=float(np.mean(np.all(np.rint(pred)==ty,axis=1)))
    return {"seed":seed,"train_samples":train_n,"test_samples":test_n,
            "background_noise_max":noise,"latent_position_mae":state_mae,
            "rounded_position_accuracy":state_round_accuracy,
            "pixel_persistence_mse":baseline_pixel_mse,
            "structured_pixel_prediction_mse":structured_pixel_mse,
            "interpretation":"oracle states from simulator; not learned visual encoder or JEPA"}

def plan(start,goal,W,horizon=10):
    """Beam-free exhaustive one-step greedy planning in learned latent coordinates."""
    pos=np.asarray(start,dtype=float)
    path=[]
    for _ in range(horizon):
        candidates=np.repeat(pos[None,:],len(ACTIONS),axis=0)
        nxt=predict(candidates,np.arange(len(ACTIONS)),W)
        # Only safe interior positions in this simple environment.
        cost=np.linalg.norm(nxt-np.asarray(goal),axis=1)
        cost[(nxt<2).any(axis=1)|(nxt>SIZE-3).any(axis=1)]=np.inf
        choice=int(np.argmin(cost))
        pos=nxt[choice];path.append({"action":choice,"position":pos.tolist(),"distance":float(cost[choice])})
        if cost[choice]<0.5: break
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');p.add_argument('--seed',type=int,default=3);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report=evaluate(seed=args.seed)
    x,a,y,_,_=make_data(1000,args.seed)
    report['planned_path']=plan([4,4],[9,7],fit_transition(x,a,y))
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
