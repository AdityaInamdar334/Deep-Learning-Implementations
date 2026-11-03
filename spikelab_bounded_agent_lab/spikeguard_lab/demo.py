import json
from .data import make_synthetic_recording
from .agent import run_policy
from .evaluator import evaluate


def main():
    r=make_synthetic_recording(7)
    for policy in ['improvised','no_verification','bounded']:
        out=run_policy(r,policy,seed=7,surrogates=9)
        grade=evaluate(r,out)
        print(f'\n{policy}: {grade["passed"]}/{grade["possible"]} evaluation checks')
        print('failed:', [k for k,v in grade['checks'].items() if not v])
        print('first pair:',out['analysis']['cue']['pair_results'][0])
        print('clarification requested:',out['clarification_requested'])

if __name__=='__main__':main()
