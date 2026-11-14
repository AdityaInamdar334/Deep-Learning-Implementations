"""Optional figure regeneration, requires matplotlib."""
import matplotlib.pyplot as plt
from pathlib import Path
from attention import scores

sizes = [8, 16, 32, 64]
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for label, keys, color in [('Spatial self-attention', [s*s for s in sizes], '#aa4a25'),
                           ('Text cross-attention', [77]*4, '#12665e')]:
    ax.plot(range(4), [scores(s, s, k) for s, k in zip(sizes, keys)],
            'o-', label=label, linewidth=2.5, color=color)
ax.set_yscale('log', base=2)
ax.set_xticks(range(4), [f'{s} × {s}' for s in sizes])
ax.set_yticks([1/16, 1/4, 1, 4, 16, 64, 256], ['1/16', '1/4', '1', '4', '16', '64', '256'])
ax.set(xlabel='Spatial grid', ylabel='Dense scores in MiB (log scale)',
       title='At the finest grid, one attention matrix is 53 times larger')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.2)
fig.savefig(Path(__file__).with_name('attention-sizes.png'), dpi=150)
