import argparse
import json
import time
from pathlib import Path
from .data import make_synthetic_recording
from .agent import run_policy
from .evaluator import evaluate


def benchmark(seeds=5,surrogates=19):
    rows=[]
    for seed in range(seeds):
        rec=make_synthetic_recording(seed=seed)
        for policy in ['improvised','silent_downsample','no_verification','bounded']:
            start=time.perf_counter()
            out=run_policy(rec,policy=policy,seed=seed,surrogates=surrogates)
            secs=time.perf_counter()-start
            ev=evaluate(rec,out)
            rows.append({'seed':seed,'policy':policy,'score':ev['passed'],
                         'score_total':ev['possible'],'elapsed_s':secs,
                         'n_tool_calls':len(out['audit_log']),
                         'clarified':out['clarification_requested'],
                         'checks':ev['checks']})
    return {'about':'Deterministic scripted workflow simulations, not LLM trials or SpikeLab replication.',
            'seeds':seeds,'surrogates':surrogates,'rows':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--surrogates',type=int,default=19)
    p.add_argument('--output',default='results/benchmark.json')
    args=p.parse_args()
    result=benchmark(args.seeds,args.surrogates)
    by_policy={}
    for policy in ['improvised','silent_downsample','no_verification','bounded']:
        rows=[x for x in result['rows'] if x['policy']==policy]
        by_policy[policy]={'mean_score':round(sum(x['score'] for x in rows)/len(rows),2),
                           'mean_runtime_s':round(sum(x['elapsed_s'] for x in rows)/len(rows),3),
                           'mean_tool_calls':round(sum(x['n_tool_calls'] for x in rows)/len(rows),2)}
    result['summary']=by_policy
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(by_policy,indent=2))
    print('Saved:',path)

if __name__=='__main__': main()
