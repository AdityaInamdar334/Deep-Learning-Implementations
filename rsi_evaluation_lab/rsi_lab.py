"""Illustrative, synthetic benchmark coverage and projection sensitivity lab.
No actual model scores or forecast estimates are used.
"""
import argparse, json, math, random
from statistics import mean

DOMAINS = ('pretraining', 'posttraining', 'harness', 'judgment', 'systems')


def apparent_score(true_scores, covered=('pretraining','posttraining','harness'), bias=0.0):
    return min(1.0, max(0.0, mean(true_scores[k] for k in covered) + bias))


def trajectory(months=18, rates=None, initial=None):
    rates = rates or {'pretraining':.030,'posttraining':.026,'harness':.035,'judgment':.009,'systems':.011}
    initial = initial or {'pretraining':.43,'posttraining':.35,'harness':.40,'judgment':.19,'systems':.22}
    return [{k:min(1.0,initial[k]+rates[k]*m) for k in DOMAINS} for m in range(months+1)]


def first_crossing(scores, cutoff=.8, coverage=DOMAINS):
    for month, record in enumerate(scores):
        if all(record[k]>=cutoff for k in coverage):
            return month
    return None


def judge_example(true_improvement=.04, judge_bias=.07):
    return {'true_improvement':true_improvement, 'estimated_improvement':true_improvement+judge_bias, 'evaluation_bias':judge_bias}


def run(seed=23):
    records=trajectory(80)
    recorded=[apparent_score(s) for s in records]
    broad=[apparent_score(s,DOMAINS) for s in records]
    out={
      'disclaimer':'SYNTHETIC educational example; not Vals/METR data or a forecast',
      'selected_domain_scores': [{'month':m,'narrow':round(recorded[m],3),'all_five':round(broad[m],3)} for m in (0,6,12,18)],
      'narrow_all_domains_above_0_8_month': first_crossing(records,.8,('pretraining','posttraining','harness')),
      'all_domains_above_0_8_month': first_crossing(records,.8),
      'judge_bias_example':judge_example(),
      'evaluation_signals': {'agent_dev_improvement':.04,'independent_holdout_improvement':-.01},
    }
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default=None)
    args=parser.parse_args()
    result=run()
    data=json.dumps(result,indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(data+'\n')
    print(data)
