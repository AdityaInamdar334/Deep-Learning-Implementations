"""Optional plots, used only when Matplotlib is installed."""
import numpy as np

def plot_examples(output_dir):
    from pathlib import Path
    from .topology import rips_h0_h1, normalize_rows, summarize
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(18)
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    blob = rng.normal(0, 0.35, size=(12, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, points, title in zip(axes, (circle, blob), ('12 points on a circle','12 Gaussian points')):
        ax.scatter(points[:,0], points[:,1], s=45)
        ax.set_aspect('equal'); ax.set_title(title); ax.grid(alpha=.2)
        d = rips_h0_h1(points)[1]
        ax.text(.03, .04, f'H1 total persistence: {summarize(d)["total"]:.2f}', transform=ax.transAxes,
                bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
    fig.suptitle('Demonstration of H1 persistence: topology depends on chosen cloud')
    fig.tight_layout(); fig.savefig(output_dir/'01_circle_vs_blob.png', dpi=160)
    plt.close(fig)
    return output_dir/'01_circle_vs_blob.png'

def plot_training(results, output_dir):
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for c, rows in results.items():
        epochs=[r['epoch'] for r in rows]
        axes[0].plot(epochs, [r['test_accuracy_on_true_labels'] for r in rows], label=c.replace('_',' '))
        axes[1].plot(epochs, [r['h1_total'] for r in rows], label=c.replace('_',' '))
    axes[0].set_ylabel('True-label test accuracy'); axes[1].set_ylabel('Normalized first-layer H1 total persistence')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Toy 2D MLP experiment: one seed; NOT paper results')
    fig.tight_layout(); fig.savefig(output_dir/'02_toy_training.png', dpi=160)
    plt.close(fig)
    return output_dir/'02_toy_training.png'
