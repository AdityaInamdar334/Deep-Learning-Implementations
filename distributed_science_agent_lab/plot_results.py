"""Generate original figures from the stored benchmark, not invented outcomes."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
report=json.loads((root/'results/benchmark.json').read_text())
s=report['summary']
(root/'figures').mkdir(exist_ok=True)
labels=['Generalist 8','Specialists 8','Serial all','Parallel all']
keys=['generalist_8','specialists_8_verified','specialists_all_serial','specialists_all_parallel']
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(labels,[s[k]['independently_replicated_claims']['mean'] for k in keys])
ax.set_ylabel('Independently replicated claims (mean)')
ax.set_title('Coverage and verification under different work budgets')
ax.tick_params(axis='x',rotation=18)
fig.tight_layout();fig.savefig(root/'figures/01_coverage.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(['Serial specialists','Parallel specialists'],
       [s[k]['runtime_seconds']['mean'] for k in ['specialists_all_serial','specialists_all_parallel']])
ax.set_ylabel('Wall-clock seconds (mean)')
ax.set_title('Simulated remote latency: same statistical work, different scheduling')
fig.tight_layout();fig.savefig(root/'figures/02_parallel_wall_clock.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
labels=['Both guards','No holdout','No BH FDR','Neither']
keys=['specialists_all_parallel','no_holdout_verification','no_multiple_testing_correction','neither_safeguard']
ax.bar(labels,[s[k]['claims_without_replication']['mean'] for k in keys])
ax.set_ylabel('Accepted claims failing independent replication (mean)')
ax.set_title('Verification and multiple testing ablations')
ax.tick_params(axis='x',rotation=14)
fig.tight_layout();fig.savefig(root/'figures/03_ablations.png',dpi=180);plt.close(fig)
print('Saved original benchmark figures in figures/')
