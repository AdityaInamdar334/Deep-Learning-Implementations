"""Illustrations accompanying the Lorenz stable-manifold article.

Run: python lorenz_lab.py --output figures
Requires numpy, scipy, matplotlib.
This is a visualization, not a rigorous manifold computation.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]

def trajectory(start, duration=35.0, samples=7000):
    times = np.linspace(0.0, duration, samples)
    solution = solve_ivp(lorenz, (0.0, duration), start, t_eval=times,
                         method='DOP853', rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T

def plot_attractor(output):
    data = trajectory([1.0, 1.0, 1.0])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*data[600:].T, lw=0.45)
    ax.set(xlabel='x', ylabel='y', zlabel='z', title='Lorenz attractor, standard parameters')
    fig.tight_layout(); fig.savefig(output / 'lorenz_attractor.png', dpi=170); plt.close(fig)

def plot_sensitivity(output):
    a = trajectory([1.0, 1.0, 1.0], duration=30)
    b = trajectory([1.0 + 1e-8, 1.0, 1.0], duration=30)
    separation = np.linalg.norm(a - b, axis=1)
    times = np.linspace(0, 30, len(separation))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(times, np.maximum(separation, 1e-16))
    ax.set(xlabel='Time (model units)', ylabel='Distance between trajectories',
           title='The butterfly effect: nearby initial conditions diverge')
    ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(output / 'butterfly_effect.png', dpi=170); plt.close(fig)

def plot_saddle(output):
    t = np.linspace(-2, 2, 500)
    fig, ax = plt.subplots(figsize=(8, 6))
    for x0, y0 in [(0.3, 1.0),(-0.3, 1.0),(0.3,-1.0),(-0.3,-1.0), (0,1),(0,-1), (0.5,0),(-0.5,0)]:
        ax.plot(x0*np.exp(t), y0*np.exp(-t),lw=1.3)
    ax.axhline(0, linestyle='--', label='Unstable axis (x)')
    ax.axvline(0, linestyle=':', label='Stable axis (y)')
    ax.set(xlim=(-3,3),ylim=(-3,3), xlabel='x',ylabel='y',title='Linear saddle: x\u2032=x, y\u2032=-y')
    ax.legend(loc='upper right'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'stable_unstable_linear.png', dpi=170);plt.close(fig)

def plot_local_eigenvectors(output):
    # Jacobian of Lorenz field at the origin
    jac=np.array([[-SIGMA,SIGMA,0],[RHO,-1,0],[0,0,-BETA]])
    eigenvalues,eigenvectors=np.linalg.eig(jac)
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111, projection='3d')
    for val, vec in zip(eigenvalues,eigenvectors.T):
        scale=8*vec/np.linalg.norm(vec)
        color='tab:red' if val.real>0 else 'tab:blue'
        ax.plot([-scale[0],scale[0]],[-scale[1],scale[1]],[-scale[2],scale[2]],
                color=color, lw=3, label=f'eigenvalue {val.real:.3f}')
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='Linearized Lorenz flow at origin: 1 unstable, 2 stable directions')
    ax.legend();fig.tight_layout();fig.savefig(output/'lorenz_local_directions.png',dpi=170);plt.close(fig)
    return eigenvalues

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='figures');args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    plot_attractor(output);plot_sensitivity(output);plot_saddle(output)
    eigenvalues=plot_local_eigenvectors(output)
    print('Jacobian eigenvalues at the origin:', np.sort(eigenvalues))
    print('Created four PNG figures in', output)

if __name__=='__main__': main()
