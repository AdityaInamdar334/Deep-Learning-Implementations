import argparse,json
from pathlib import Path

def estimate(steps,coverage,input_tokens,output_tokens,final_input,final_output,local_input):
    base=steps*(input_tokens+output_tokens)+final_input+final_output
    hybrid=steps*(1-coverage)*(input_tokens+output_tokens)+final_input+final_output
    return {'baseline_frontier_tokens':base,'hybrid_frontier_tokens':hybrid,'frontier_reduction':1-hybrid/base,'local_input_tokens':steps*local_input,'total_hybrid_tokens':hybrid+steps*local_input,'note':'Analytical scenario, not measured token billing or latency.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--coverage',type=float,default=.72); p.add_argument('--local-input',type=int,default=300); a=p.parse_args(); assert 0<=a.coverage<=1 and a.local_input>=0; r=estimate(4,a.coverage,800,80,800,160,a.local_input); Path('results/cost.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
