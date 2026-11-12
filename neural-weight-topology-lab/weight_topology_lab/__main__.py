import argparse
import json
from pathlib import Path
from .topology import rips_h0_h1, summarize
from .plotting import plot_examples

def main():
    p=argparse.ArgumentParser(description='CPU-friendly neural weight topology tutorial')
    p.add_argument('command',choices=['shapes','demo','inspect'])
    p.add_argument('--output',default='results')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cloud',default='',help='CSV matrix rows=neurons for inspect')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.command == 'shapes':
        plot_examples(out)
        print(f'Saved example: {out}/01_circle_vs_blob.png')
    elif args.command == 'demo':
        from .experiment import training_run
        from .plotting import plot_training
        plot_examples(out)
        r=training_run(seed=args.seed, epochs=args.epochs)
        (out/'toy_metrics.json').write_text(json.dumps(r,indent=2))
        plot_training(r,out)
        print('Toy final results (not paper results):')
        for k,rows in r.items():
            v=rows[-1]
            print(f'{k:>17}: true-test={v["test_accuracy_on_true_labels"]:.3f}, H1-total={v["h1_total"]:.4f}')
        print(f'Files saved under {out}')
    else:
        if not args.cloud: p.error('--cloud required for inspect')
        import numpy as np
        from .topology import normalize_rows
        a=np.loadtxt(args.cloud,delimiter=',')
        if a.shape[0]>16: p.error('inspect demo supports at most 16 rows; use Ripser.py for larger clouds')
        for name,pts in [('raw',a),('normalized',normalize_rows(a))]:
            ds=rips_h0_h1(pts)
            print(name, {f'H{k}':summarize(ds[k]) for k in [0,1]})
if __name__=='__main__': main()
