"""Small, deterministic 2D imitation-learning and deployment-feedback simulation."""
from __future__ import annotations
import argparse
import json
import numpy as np

DT=0.2
GAIN=1.25
LIMIT=0.35

def expert_action(state, target):
    return np.clip(GAIN*(np.asarray(target)-np.asarray(state)), -LIMIT, LIMIT)

def collect_demonstrations(n=600, seed=17):
    rng=np.random.default_rng(seed)
    states=rng.uniform(-1.5, 1.5, (n, 2))
    # Demos sampled near targets so the expert action is usually unsaturated.
    targets=states + rng.uniform(-0.22, 0.22, (n, 2))
    inputs=np.column_stack((targets-states, np.ones(n)))
    actions=np.array([expert_action(s,t) for s,t in zip(states, targets)])
    return inputs, actions

def fit_policy(X,Y,ridge=1e-3):
    # Least squares behavioural cloning, not a diffusion model.
    eye=np.eye(X.shape[1]); eye[-1,-1]=0.0
    return np.linalg.solve(X.T@X + ridge*eye, X.T@Y)

def predict_action(weights,state,target):
    diff=np.asarray(target)-np.asarray(state)
    return np.clip(np.array([*diff,1.])@weights, -LIMIT,LIMIT)

def rollout(weights=None, mode='feedback', perturb=True, horizon=65, seed=4):
    rng=np.random.default_rng(seed)
    state=np.array([-1.,-0.7]); target=np.array([1.,0.9]); trajectory=[state.copy()]
    initial=expert_action(state,target)
    for step in range(horizon):
        if mode=='open_loop':
            action=initial
        elif mode=='expert':
            action=expert_action(state,target)
        elif mode=='feedback':
            if weights is None: raise ValueError('weights needed')
            action=predict_action(weights,state,target)
        else: raise ValueError('unsupported mode')
        state=state+DT*action
        if perturb and step==22:
            # An unexpected displacement: analogous to moved object / external push.
            state+=np.array([-0.55,0.25])
        trajectory.append(state.copy())
    traj=np.array(trajectory)
    return {'mode':mode,'final_error':float(np.linalg.norm(traj[-1]-target)),
            'success':bool(np.linalg.norm(traj[-1]-target)<0.13), 'path':traj,'target':target}

def compare(seed=17):
    X,Y=collect_demonstrations(seed=seed)
    W=fit_policy(X,Y)
    cases=[rollout(W,'open_loop'),rollout(W,'feedback'),rollout(W,'expert')]
    return W,cases

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plot',action='store_true'); p.add_argument('--json',action='store_true'); args=p.parse_args()
    W,cases=compare()
    if args.json:
        print(json.dumps([{k:v for k,v in r.items() if k not in ('path','target')} for r in cases],indent=2))
    else:
        for r in cases: print(f"{r['mode']:>10}: final error={r['final_error']:.3f}, success={r['success']}")
    if args.plot:
        try: import matplotlib.pyplot as plt
        except ImportError: raise SystemExit('Install matplotlib for --plot')
        for r in cases: plt.plot(r['path'][:,0],r['path'][:,1],label=r['mode'])
        plt.scatter([1],[0.9],marker='*',s=160,label='target')
        plt.scatter([-1],[-0.7],marker='o',label='start')
        plt.xlabel('x position'); plt.ylabel('y position'); plt.title('Imitation policy vs open-loop replay after disturbance')
        plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.savefig('deployment_feedback.png',dpi=170)
        print('Saved deployment_feedback.png')
if __name__=='__main__': main()
