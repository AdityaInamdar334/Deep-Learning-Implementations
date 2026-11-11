"""Command line: python -m option_audit demo | analyze PATH | train"""
import argparse
import json
from pathlib import Path
from .core import make_toy_questions,toy_audit,identical_options_slot_prior,audit_records
from .train import make_rows,train,eval_model


def render_bar(metrics,path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    keys=list(metrics)
    vals=[metrics[k]['flip_rate']*100 for k in keys]
    fig, ax=plt.subplots(figsize=(8,4.5))
    ax.bar(keys,vals)
    ax.set_ylabel('Questions with a changed answer (%)')
    ax.set_title('Synthetic position-bias experiment (NOT paper results)')
    ax.set_ylim(0,max(vals)*1.2+1)
    ax.tick_params(axis='x',labelrotation=15)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    return True


def main():
    p=argparse.ArgumentParser(description='Order sensitivity audit; pure Python, no downloaded model required')
    sub=p.add_subparsers(dest='cmd',required=True)
    demo=sub.add_parser('demo',help='generate synthetic questions and audit position bias')
    demo.add_argument('--out',default='results')
    analyze=sub.add_parser('analyze',help='analyze external model results in JSONL format')
    analyze.add_argument('jsonl')
    analyze.add_argument('--out',default='results/external_audit.json')
    training=sub.add_parser('train',help='synthetic matched-data shuffle/no-shuffle ablation')
    training.add_argument('--out',default='results/training_ablation.json')
    args=p.parse_args()
    if args.cmd=='demo':
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        questions=make_toy_questions()
        metrics={key:toy_audit(questions,strength) for key,strength in [('position_biased',.85),('reduced_bias',.20),('parallel_independent',0.0)]}
        compact={key:{k:v for k,v in value.items() if k!='details'} for key,value in metrics.items()}
        compact['identical_options_K12']={key:identical_options_slot_prior(12,strength) for key,strength in [('position_biased',.85),('parallel_independent',0.0)]}
        (out/'toy_metrics.json').write_text(json.dumps(compact,indent=2)+'\n')
        render_bar(metrics,out/'toy_flip_rates.png')
        print(json.dumps({k:{'flip_rate':v['flip_rate'],'mean_accuracy':v['mean_accuracy']} for k,v in metrics.items()},indent=2))
        print(f'Wrote results to {out}; all numbers are SYNTHETIC, not Maya/Laya reproduction.')
    elif args.cmd=='analyze':
        records=[json.loads(x) for x in Path(args.jsonl).read_text().splitlines() if x.strip()]
        result=audit_records(records)
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
    else:
        rows=make_rows()
        fixed=train(rows,shuffle_options=False)
        shuffled=train(rows,shuffle_options=True)
        result={'note':'Synthetic position-confounded training set, NOT a Laya model or Maya paper replication',
                'no_shuffle':{'params':fixed, 'evaluation':eval_model(fixed)},
                'shuffle':{'params':shuffled, 'evaluation':eval_model(shuffled)}}
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
