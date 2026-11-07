"""Plot a benchmark CSV; requires matplotlib."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='results/moe_benchmark.csv')
    p.add_argument('--out', default='results/moe_lab_plot.png')
    a = p.parse_args()
    with open(a.csv, newline='') as f:
        rows = list(csv.DictReader(f))
    xs = [int(r['top_k']) for r in rows]
    macs = [int(r['estimated_expert_and_router_MACs_per_token']) / 1e6 for r in rows]
    times = [float(r['elapsed_ms_CPU_median']) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, macs, marker='o', label='Theoretical MACs/token (millions)')
    ax.set_xlabel('Selected experts per token (top-k)')
    ax.set_ylabel('MACs per token (millions)')
    other = ax.twinx()
    other.plot(xs, times, marker='s', linestyle='--', label='Toy CPU elapsed ms')
    other.set_ylabel('Elapsed ms, median CPU measurement')
    ax.set_title('Sparse MoE: fewer theoretical MACs vs dispatch overhead')
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = other.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, loc='upper left')
    fig.tight_layout()
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f'Plot saved: {output}')


if __name__ == '__main__':
    main()
