"""Learn an approximate forced Van der Pol solution operator using a tiny DeepONet.

This is a discretized branch/trunk neural operator demonstration, NOT an FNO.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn

MU = 1.0
T_END = 6.0
GRID = np.linspace(0, T_END, 41)


def forcing(t, coeffs):
    """Smooth family with several independent sine/cosine coefficients."""
    a, b, c, d = coeffs
    return (a * np.sin(0.8*t) + b * np.cos(1.3*t)
            + c * np.sin(2.1*t) + d * np.cos(0.45*t))


def vector_field(t, state, coeffs=None, mu=MU):
    x, v = state
    f = 0.0 if coeffs is None else forcing(t, coeffs)
    return np.array([v, mu * (1-x*x) * v - x + f], dtype=float)


def solve_trajectory(coeffs, initial=(0.5, 0.0), t_grid=GRID, mu=MU):
    t_grid = np.asarray(t_grid, dtype=float)
    result = solve_ivp(lambda t, z: vector_field(t, z, coeffs, mu),
                       (float(t_grid[0]), float(t_grid[-1])), initial,
                       t_eval=t_grid, rtol=1e-8, atol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    inputs, outputs = [], []
    for _ in range(n):
        coeffs = rng.uniform(-0.6, 0.6, size=4)
        x0 = rng.uniform(-1.2, 1.2)
        v0 = rng.uniform(-0.8, 0.8)
        trajectory = solve_trajectory(coeffs, (x0, v0))
        sample = np.concatenate([forcing(GRID, coeffs), [x0, v0]])
        inputs.append(sample)
        outputs.append(trajectory[:, 0])
    return (torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(outputs), dtype=torch.float32))


class DeepONet(nn.Module):
    """Branch encodes forcing samples+initial state; trunk encodes query time."""
    def __init__(self, sensors=len(GRID), width=64, latent=64):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(sensors+2, width), nn.Tanh(),
                                    nn.Linear(width, width), nn.Tanh(),
                                    nn.Linear(width, latent))
        self.trunk = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                   nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, latent))
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = latent ** -0.5

    def forward(self, sampled_functions, query_times):
        """Return [batch, queries] predictions; query times are in [0,1]."""
        b = self.branch(sampled_functions)
        t = self.trunk(query_times.reshape(-1, 1))
        return b @ t.T * self.scale + self.bias


def draw_phase_portrait(out_dir):
    xs = np.linspace(-2.6, 2.6, 21)
    vs = np.linspace(-3.0, 3.0, 21)
    X, V = np.meshgrid(xs, vs)
    dX = V
    dV = (1-X*X)*V-X
    norm = np.maximum(np.sqrt(dX*dX+dV*dV), 1e-9)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.quiver(X, V, dX/norm, dV/norm, alpha=0.45, color='#5d6c88')
    for init in [(0.2,0.0), (2.4,0.3), (-1.5,1.5), (0,0)]:
        trajectory = solve_trajectory(None, init, np.linspace(0, 40, 1601))
        ax.plot(trajectory[:,0],trajectory[:,1],label=f'initial={init}',lw=1.7)
        ax.scatter(*init, s=22)
    ax.set(xlabel='Displacement x',ylabel='Velocity dx/dt',title='Unforced Van der Pol: phase portrait and limit cycle')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir/'phase_portrait.png',dpi=150);plt.close(fig)


def run(out_dir: Path, n_train=192, n_test=40, epochs=180, seed=7):
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    train_x, train_y = make_dataset(n_train, seed)
    test_x, test_y = make_dataset(n_test, seed+1000)
    model=DeepONet()
    optim=torch.optim.Adam(model.parameters(),lr=0.002)
    times=torch.tensor((GRID/T_END).reshape(-1,1),dtype=torch.float32)
    history=[]
    for epoch in range(epochs):
        model.train()
        perm=torch.randperm(n_train)
        for ids in perm.split(32):
            optim.zero_grad()
            pred=model(train_x[ids],times)
            loss=((pred-train_y[ids])**2).mean()
            loss.backward(); optim.step()
        if epoch % 10 == 0 or epoch==epochs-1:
            model.eval()
            with torch.no_grad():
                tr=((model(train_x,times)-train_y)**2).mean().item()
                te=((model(test_x,times)-test_y)**2).mean().item()
            history.append({'epoch':epoch+1,'train_mse':round(tr,6),'test_mse':round(te,6)})
            print(f'epoch {epoch+1:>3}: train MSE={tr:.5f}; held-out MSE={te:.5f}')
    model.eval()
    with torch.no_grad():
        prediction=model(test_x,times).numpy()
    true=test_y.numpy()
    rmse=np.sqrt(np.mean((prediction-true)**2,axis=1))
    global_rmse=float(np.sqrt(np.mean((prediction-true)**2)))
    baseline=float(np.sqrt(np.mean((true-true.mean(axis=0,keepdims=True))**2)))
    draw_phase_portrait(out_dir)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,ax in enumerate(axes.flat):
        ax.plot(GRID,true[i],label='ODE numerical solution',lw=2)
        ax.plot(GRID,prediction[i],label='DeepONet prediction',lw=2,ls='--')
        ax.set(xlabel='Time',ylabel='Displacement',title=f'Unseen forcing {i+1}: RMSE {rmse[i]:.3f}')
        ax.grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(out_dir/'heldout_predictions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot([r['epoch'] for r in history],[r['train_mse'] for r in history],label='Train MSE')
    ax.plot([r['epoch'] for r in history],[r['test_mse'] for r in history],label='Held-out MSE')
    ax.set(xlabel='Epoch',ylabel='MSE',title='Operator learning curves');ax.set_yscale('log');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out_dir/'learning_curves.png',dpi=150);plt.close(fig)
    with (out_dir/'training_history.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['epoch','train_mse','test_mse']);writer.writeheader();writer.writerows(history)
    metrics={'mu':MU,'time_domain':[0,T_END], 'time_samples':len(GRID), 'n_train':n_train,'n_test':n_test,
             'epochs':epochs,'heldout_rmse':round(global_rmse,6),'mean_curve_baseline_rmse':round(baseline,6),
             'architecture':'DeepONet branch/trunk, NOT a Fourier Neural Operator',
             'heldout_first_four_rmse':[round(float(x),6) for x in rmse[:4]],
             'disclaimer':'Fixed 41-sensor training grid; no discretization invariance demonstrated.'}
    (out_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    torch.save(model.state_dict(),out_dir/'deeponet_weights.pt')
    return metrics


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('figures'))
    parser.add_argument('--epochs',type=int,default=180)
    parser.add_argument('--train',type=int,default=192)
    parser.add_argument('--test',type=int,default=40)
    args=parser.parse_args()
    print(json.dumps(run(args.out,args.train,args.test,args.epochs),indent=2))
