"""Five-seed bounded, synthetic weight inspection and repair benchmark."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import torch
from .core import make_policy, permute_hidden, inspect_axis_naive, inspect_axis_invariant, compile_xor
from .editing import propose_edit,diff_audit
from .evaluate import classification_accuracy,navigation_success

POLICIES=['original','semantic','half_semantic','top_magnitude','random_two','global_shrink','wrong_path']

def trial(seed, *, readings=60, samples=3000, episodes=300):
    gen=torch.Generator().manual_seed(seed+100)
    correct_naive=0; correct_invariant=0; checked=0
    for axis in range(3):
        for j in range(readings):
            m=make_policy(axis=axis,nuisance=3)
            pm=permute_hidden(m,torch.randperm(m.fc1.out_features,generator=gen))
            correct_naive += inspect_axis_naive(pm)==axis
            correct_invariant += inspect_axis_invariant(pm)==axis
            checked+=1
    policies={}
    original=make_policy(broken=True)
    for i,method in enumerate(POLICIES):
        p=propose_edit(original,method,seed=seed*101+17)
        audit=diff_audit(original,p)
        seen=classification_accuracy(p,seed=seed*503+200,n=samples)
        shifted=classification_accuracy(p,seed=seed*503+201,n=samples,shift=2.5)
        nav=navigation_success(p,seed=seed*503+202,episodes=episodes,horizon=14)
        policies[method]={'accuracy':seen['accuracy'],'shift_accuracy':shifted['accuracy'], 'stall_rate':seen['stall_rate'], 'shift_stall_rate':shifted['stall_rate'], 'navigation_success':nav['success_rate'], 'changed_weights':audit['changed'],'changed_fraction':audit['fraction']}
    return {'seed':seed,'readers':{'naive':correct_naive/checked,'permutation_invariant':correct_invariant/checked},'replicas':checked, 'write_xor_correct':bool(torch.equal(compile_xor()(torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])).round().flatten().long(),torch.tensor([0,1,1,0]))),'policies':policies}

def run(seeds=5,readings=60,samples=3000,episodes=300):
    trials=[trial(i,readings=readings,samples=samples,episodes=episodes) for i in range(seeds)]
    keys=list(trials[0]['policies'][POLICIES[0]])
    mean=lambda xs:float(statistics.mean(xs))
    sd=lambda xs:float(statistics.stdev(xs)) if len(xs)>1 else 0.0
    summary={'readers':{m:{'mean':mean([t['readers'][m] for t in trials]),'sample_sd':sd([t['readers'][m] for t in trials])} for m in ['naive','permutation_invariant']}, 'policies':{p:{k:{'mean':mean([t['policies'][p][k] for t in trials]),'sample_sd':sd([t['policies'][p][k] for t in trials])} for k in keys} for p in POLICIES}}
    return {'metadata':{'independent_synthetic_benchmark':True, 'not_an_llm_experiment':True,'seeds':seeds,'readings_per_axis_per_seed':readings,'holdout_samples_per_policy':samples,'navigation_episodes_per_policy':episodes,'random_seed_start':0,'architectural_prior': 'Know input coordinate meaning and action index 2 means stall', 'no_gradient_edits': True},'summary':summary,'trials':trials}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--readings',type=int,default=60);p.add_argument('--samples',type=int,default=3000);p.add_argument('--episodes',type=int,default=300);p.add_argument('--output',default='results/benchmark.json');a=p.parse_args()
    result=run(a.seeds,a.readings,a.samples,a.episodes)
    out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',out)
    print('READERS', {k:round(v['mean'],4) for k,v in result['summary']['readers'].items()})
    for k,v in result['summary']['policies'].items():
        print(k,'accuracy',round(v['accuracy']['mean'],4),'OOD',round(v['shift_accuracy']['mean'],4),'nav',round(v['navigation_success']['mean'],4),'changed',round(v['changed_weights']['mean']))
if __name__=='__main__':main()
