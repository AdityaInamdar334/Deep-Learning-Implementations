"""Summarize actual results and plot heuristic baselines after a complete run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    raw=json.loads((ROOT/'results/results.json').read_text()); rows=raw['runs']; baselines=json.loads((ROOT/'results/baselines.json').read_text()); names=list(dict.fromkeys(r['name'] for r in rows)); summary={}
    for name in names:
        rs=[r for r in rows if r['name']==name]; k='1' if name.startswith('dense') else '3'; by_hop=[float(np.mean([r['evaluation'][k][str(h)]['accuracy'] for r in rs])) for h in range(1,7)]; summary[name]={'parameters':rs[0]['parameters'],'train_range_accuracy':float(np.mean(by_hop[:3])),'longer_hop_accuracy':float(np.mean(by_hop[3:])),'by_hop':by_hop,'seed_count':len(rs),'mean_training_seconds':float(np.mean([r['train_seconds'] for r in rs])),'mean_layer_applications_per_batch_sum':float(np.mean([r['layer_applications_per_batch_sum'] for r in rs]))}
        if not name.startswith('dense'):
            summary[name]['loop_sweep_train_range']=[float(np.mean([r['evaluation'][str(l)][str(h)]['accuracy'] for r in rs for h in (1,2,3)])) for l in range(1,7)]
            summary[name]['loop_sweep_longer_range']=[float(np.mean([r['evaluation'][str(l)][str(h)]['accuracy'] for r in rs for h in (4,5,6)])) for l in range(1,7)]
    for name in ['uniform_expected','start_node','one_hop','oracle']:
        by_hop=[float(np.mean([r[name] for r in baselines if r['hops']==h])) for h in range(1,7)]; summary[name]={'by_hop':by_hop,'train_range_accuracy':float(np.mean(by_hop[:3])),'longer_hop_accuracy':float(np.mean(by_hop[3:]))}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ['dense2','dense6','loop_fixed','loop_variable','loop_moe','one_hop','start_node']:
        ax.plot(range(1,7),summary[name]['by_hop'],marker='o',ls='--' if name in ['one_hop','start_node'] else '-',label=name)
    ax.axvline(3.5,color='gray',ls=':'); ax.set(xlabel='Pointer hops (train: 1–3)',ylabel='Mean held-out accuracy',ylim=(0,1)); ax.legend(ncol=2,fontsize=9); fig.tight_layout(); fig.savefig(ROOT/'figures/heuristic_comparison.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
