"""Reproducible representation-learning demo: concentric rings and XOR."""
from pathlib import Path
import argparse
import json
import numpy as np


def make_rings(n=1600, seed=42):
    rng = np.random.default_rng(seed)
    # Stratify the concentric rings to avoid finite-sample angular imbalance.
    y = np.arange(n) % 2
    theta = np.empty(n, dtype=float)
    for cls in (0, 1):
        indices = np.flatnonzero(y == cls)
        theta[indices] = (np.arange(len(indices)) * 2*np.pi / len(indices)
                          + rng.uniform(0, 2*np.pi))
    radius = np.where(y == 0, 1.0, 2.2) + rng.normal(0, 0.12, size=n)
    x = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])
    order = rng.permutation(n)
    return x[order], y[order]


def radial_features(x):
    return np.sum(x*x, axis=1, keepdims=True)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def fit_logistic(x, y, steps=3000, lr=0.15):
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-8
    z = (x - mean) / scale
    w = np.zeros(z.shape[1])
    b = 0.0
    history = []
    for i in range(steps):
        p = sigmoid(z @ w + b)
        error = p - y
        w -= lr * (z.T @ error / len(y))
        b -= lr * error.mean()
        if i % 100 == 0:
            history.append(float(-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).mean()))
    return dict(w=w, b=b, mean=mean, scale=scale, history=history)


def predict(model, x):
    return (sigmoid(((x-model['mean']) / model['scale']) @ model['w'] + model['b']) >= 0.5).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='figures')
    a = parser.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x_train, y_train = make_rings(1400, 43)
    x_test, y_test = make_rings(600, 2026)
    plain = fit_logistic(x_train, y_train)
    radial = fit_logistic(radial_features(x_train), y_train)
    raw_acc = float(np.mean(predict(plain, x_test)==y_test))
    radial_acc = float(np.mean(predict(radial, radial_features(x_test))==y_test))
    metrics = dict(train_samples=len(y_train),test_samples=len(y_test),raw_linear_test_accuracy=raw_acc,radial_linear_test_accuracy=radial_acc,seed_train=43,seed_test=2026)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # User requested no colors; stick to matplotlib defaults.
    fig, ax = plt.subplots(1,2,figsize=(11,4.7))
    for cls in (0,1):
        m = y_test==cls
        ax[0].scatter(x_test[m,0],x_test[m,1],s=9,alpha=.65,label=f'Class {cls}')
    ax[0].set_title('Original Cartesian inputs: (x, y)')
    ax[0].set_xlabel('x');ax[0].set_ylabel('y'); ax[0].axis('equal');ax[0].legend()
    squared_radius = radial_features(x_test)[:,0]
    for cls in (0,1):
        m=y_test==cls
        ax[1].scatter(squared_radius[m], np.full(m.sum(),cls),s=9,alpha=.55,label=f'Class {cls}')
    ax[1].set_title('Transformed input: r² = x² + y²')
    ax[1].set_xlabel('r²');ax[1].set_ylabel('Class');ax[1].set_yticks([0,1]);ax[1].legend()
    fig.suptitle('A feature transform can simplify a classification problem')
    fig.tight_layout();fig.savefig(out/'representation_shift.png',dpi=180);plt.close(fig)
    grid=np.linspace(-3,3,300)
    xx,yy=np.meshgrid(grid,grid)
    pts=np.column_stack([xx.ravel(),yy.ravel()])
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    for title,model,transform,ac,ax in [('Raw x,y: straight boundary',plain,lambda x:x,raw_acc,axes[0]),('With r²: circular boundary',radial,radial_features,radial_acc,axes[1])]:
        preds=predict(model,transform(pts)).reshape(xx.shape)
        ax.contour(xx,yy,preds,levels=[.5],linewidths=2)
        for cls in (0,1):
            m=y_test==cls
            ax.scatter(x_test[m,0],x_test[m,1],s=8,alpha=.45,label=f'Class {cls}')
        ax.set_title(f'{title}\nHeld-out accuracy {ac:.1%}')
        ax.set_xlabel('x');ax.set_ylabel('y');ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(out/'decision_boundaries.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    main()
