"""Exact binary votes, within-prompt robust loss, and selected replay cases."""
from collections import defaultdict
from math import comb, exp, log, log1p, expm1
from statistics import mean

def majority_correct(p,n):
    if not 0 <= p <= 1 or n<1 or n%2==0:
        raise ValueError("Use a probability in [0,1] and a positive odd rollout count.")
    return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(n//2+1,n+1))

def pair_objective(margins,kappa,tau=.1,base=2.0):
    """Return loss, tilt weights, derivative w.r.t. each log-ratio margin g (Eq. 27–31)."""
    if not margins or not 0<kappa<=1 or min(tau,base)<=0: raise ValueError("Invalid group or temperature")
    beta=base*kappa
    z=[tau*g for g in margins]
    h=[-max(-v,0)-log1p(exp(-abs(v))) for v in z]
    logits=[v/beta for v in h]
    shift=max(logits)
    raw=[exp(v-shift) for v in logits]
    weights=[v/sum(raw) for v in raw]
    loss=-beta*(shift+log(mean(raw)))
    gradient=[-tau*w*(-expm1(v)) for w,v in zip(weights,h)]
    return loss,weights,gradient

def select(answers,scores=None):
    totals=defaultdict(float)
    if not answers or (scores is not None and len(scores)!=len(answers)):
        raise ValueError("Need one score per rollout")
    for i,answer in enumerate(answers):
        totals[answer]+=1 if scores is None else scores[i]
    return min(totals,key=lambda a:(-totals[a],a))

def replay_decision(cache,key,answers):
    """Exercise the frequency-vote replay trigger; cache rollout IDs here as stand-ins for texts."""
    if not answers: raise ValueError("Empty group")
    mode=select(answers)
    if key in cache and cache[key][0]!=mode: return "replay cached completions"
    if len(set(answers))==1:
        cache.setdefault(key,(mode,tuple(enumerate(answers))))
        return "unanimous: skip fresh pairs"
    return "use fresh pairs"

if __name__=="__main__":
    for n in (1,9,33,65):
        print(f"{n:2d} rollouts | p=.60: {majority_correct(.6,n):.4f} | p=.40: {majority_correct(.4,n):.4f}")
    margins=[log(v/(1-v))/.1 for v in (.9,.6,.2)]
    for kappa in (18/32,28/32):
        loss,w,g=pair_objective(margins,kappa)
        print(f"kappa={kappa:.4f}: weights {[round(x,3) for x in w]}, residual {sum(abs(x) for x in g)/.1:.3f}")
    a=["A","A","A","B","B"]
    print("Frequency winner:",select(a),"trajectory-score winner:",select(a,[.1,.1,.1,.6,.6]))
    cache={}
    for answers in (["A"]*4,["B","B","B","A"]):
        print(replay_decision(cache,"example",answers))
