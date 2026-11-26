"""Reproducible educational examples from Deep Learning Ch. 4. NumPy only for math."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def naive_softmax(x):
    a = np.exp(np.asarray(x, dtype=np.float64))
    return a / a.sum(axis=-1, keepdims=True)


def stable_logsumexp(x, axis=-1, keepdims=False):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x - stable_logsumexp(x, keepdims=True))


def stable_log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return x - stable_logsumexp(x, keepdims=True)


def quadratic_loss(x, hessian):
    return 0.5 * float(x @ hessian @ x)


def gradient_descent(x0, hessian, learning_rate=0.01, steps=150):
    x = np.asarray(x0, dtype=float).copy()
    path = [x.copy()]
    for _ in range(steps):
        x -= learning_rate * (hessian @ x)
        path.append(x.copy())
    return np.asarray(path)


def least_squares_on_ball(a, b, radius=1.0):
    """Solve min 0.5||Ax-b||^2 s.t. ||x||<=radius for full column rank A."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if radius <= 0:
        raise ValueError('radius must be positive')
    gram, rhs = a.T @ a, a.T @ b
    x_unconstrained = np.linalg.lstsq(a, b, rcond=None)[0]
    if np.linalg.norm(x_unconstrained) <= radius:
        return x_unconstrained, 0.0
    def x_at(lam):
        return np.linalg.solve(gram + 2 * lam * np.eye(a.shape[1]), rhs)
    low, high = 0.0, 1.0
    while np.linalg.norm(x_at(high)) > radius:
        high *= 2
    for _ in range(90):
        mid = (low + high) / 2
        if np.linalg.norm(x_at(mid)) > radius:
            low = mid
        else:
            high = mid
    return x_at(high), high


def experiment(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    logits = np.array([1000., 999., 998.])
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        bad = naive_softmax(logits)
        naive_logprob = np.log(stable_softmax(np.array([0., -1000.])))
    safe = stable_softmax(logits)
    results['softmax'] = {'logits': logits.tolist(), 'naive': [None if not np.isfinite(v) else float(v) for v in bad], 'stable':safe.tolist(), 'stable_log_probability_minus_1000':float(stable_log_softmax([0.,-1000.])[1]), 'naive_log_probability_minus_1000': str(naive_logprob[1])}
    xs = np.array([0, 1, 2, 4, 10, 50, 100, 500, 1000], dtype=float)
    scores = []
    for magnitude in xs:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            nv = naive_softmax([magnitude, magnitude-1])[0]
        st = stable_softmax([magnitude, magnitude-1])[0]
        scores.append((float(nv), float(st)))
    fig, ax = plt.subplots(figsize=(8,4.4))
    ax.plot(xs, [v[1] for v in scores], marker='o', label='Stable softmax')
    ax.plot(xs, [v[0] for v in scores], marker='x', linestyle='--', label='Naive softmax')
    ax.set(xlabel='Maximum logit', ylabel='Probability of largest logit', title='Identical mathematics, different numerical behavior', ylim=(0.5, 0.95))
    ax.legend(); fig.tight_layout(); fig.savefig(out/'softmax_stability.png', dpi=160); plt.close(fig)

    # float32 absorption and catastrophic cancellation
    a32 = np.float32(1e8)
    b32 = np.float32(1)
    results['float32'] = {'1e8_plus_1_minus_1e8':float((a32+b32)-a32), 'float64_equivalent':float((np.float64(1e8)+1)-1e8),'float32_epsilon':float(np.finfo(np.float32).eps)}
    fig, ax = plt.subplots(figsize=(8,4.4))
    magnitudes = np.logspace(0, 10, 70)
    errors = [abs(float((np.float32(v)+np.float32(1))-np.float32(v))-1) for v in magnitudes]
    ax.semilogx(magnitudes, errors)
    ax.set(xlabel='Magnitude of a in float32',ylabel='Absolute error in (a + 1) - a',title='When small increments disappear in float32')
    fig.tight_layout();fig.savefig(out/'floating_point_precision.png',dpi=160);plt.close(fig)

    # badly conditioned quadratic
    H = np.array([[80., 0.], [0., 1.]])
    x0 = np.array([3., 3.])
    paths = {'slow (η=0.005)': gradient_descent(x0,H,.005,140), 'near threshold (η=0.024)':gradient_descent(x0,H,.024,140)}
    results['optimization'] = {'condition_number':float(np.linalg.cond(H)), 'start_loss':quadratic_loss(x0,H),'final_losses':{k:quadratic_loss(p[-1],H) for k,p in paths.items()},'stability_bound_eta_less_than':2/80,'newton_one_step_loss': quadratic_loss(x0 - np.linalg.solve(H,H@x0),H)}
    grid = np.linspace(-3.2,3.2,170); xx,yy=np.meshgrid(grid,grid)
    zz=.5*(80*xx**2+yy**2)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.contour(xx,yy,zz,levels=np.geomspace(.05,400,17))
    for k,p in paths.items():
        ax.plot(p[:40,0],p[:40,1],marker='.',markersize=2,label=k)
    ax.plot(0,0,marker='*',markersize=11,label='Minimum')
    ax.set(xlabel='x₁ (steep curvature)',ylabel='x₂ (shallow curvature)',title='Learning rate and an ill-conditioned loss landscape',xlim=(-3.2,3.2),ylim=(-3.2,3.2))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'gradient_descent_canyon.png',dpi=160);plt.close(fig)

    # constrained least squares x target outside ball
    A=np.array([[2.,.3],[0.,1.5],[1.,.7]])
    target=np.array([2.,1.]); b=A@target
    x,lam=least_squares_on_ball(A,b,1.0)
    unconstrained=np.linalg.lstsq(A,b,rcond=None)[0]
    residual=A.T@(A@x-b)+2*lam*x
    results['constrained_least_squares']={'unconstrained_solution':unconstrained.tolist(),'constrained_solution':x.tolist(),'norm':float(np.linalg.norm(x)),'lambda':float(lam),'stationarity_residual_norm':float(np.linalg.norm(residual)),'complementary_slackness':float(lam*(x@x-1))}
    theta=np.linspace(0,2*np.pi,500)
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(np.cos(theta),np.sin(theta),label='Feasible boundary ||x||=1')
    ax.scatter(*unconstrained,label='Unconstrained optimum',s=65)
    ax.scatter(*x,label='Constrained optimum',s=65)
    ax.set(xlim=(-1.4,2.4),ylim=(-1.4,2.0),xlabel='x₁',ylabel='x₂',title='Constrained least squares: optimum on the boundary',aspect='equal')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'constrained_optimization.png',dpi=160);plt.close(fig)

    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results,indent=2))
    return results

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='figures')
    args=parser.parse_args()
    experiment(args.out)
