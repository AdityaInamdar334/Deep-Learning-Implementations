"""Driven, damped oscillator: analytic Laplace modes and independent IVP check."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Oscillator:
    m: float = 1.0
    c: float = 0.6
    k: float = 4.0
    force: float = 1.0
    omega: float = 1.2
    x0: float = 0.0
    v0: float = 0.0
    def validate(self):
        if self.m <= 0 or self.k <= 0 or self.c < 0 or self.omega < 0:
            raise ValueError('Require m>0, k>0, c>=0 and omega>=0')
        if self.c**2 >= 4*self.m*self.k:
            raise ValueError('This educational formula requires underdamping c² < 4mk')


def modes(p: Oscillator):
    p.validate()
    roots = np.roots([p.m,p.c,p.k])
    return np.concatenate((roots, np.array([1j*p.omega,-1j*p.omega])))


def coefficients(p: Oscillator):
    p.validate()
    alpha = p.c/(2*p.m)
    wd = np.sqrt(p.k/p.m-alpha**2)
    delta = p.k - p.m*p.omega**2
    denom = np.hypot(delta,p.c*p.omega)
    if denom == 0:
        raise ValueError('Undamped exact resonance: no bounded steady state')
    A = p.force / denom
    phi = np.arctan2(p.c*p.omega, delta)
    C = p.x0 - A*np.cos(phi)
    D = (p.v0 - A*p.omega*np.sin(phi) + alpha*C)/wd
    return A,phi,alpha,wd,C,D


def components(t,p: Oscillator):
    t = np.asarray(t,dtype=float)
    A,phi,alpha,wd,C,D = coefficients(p)
    ss = A*np.cos(p.omega*t-phi)
    tr = np.exp(-alpha*t)*(C*np.cos(wd*t)+D*np.sin(wd*t))
    return tr,ss,tr+ss


def numerical_solution(t,p: Oscillator):
    p.validate()
    def rhs(time,state):
        x,v = state
        return [v,(p.force*np.cos(p.omega*time)-p.c*v-p.k*x)/p.m]
    res=solve_ivp(rhs,(float(t[0]),float(t[-1])),[p.x0,p.v0],t_eval=t,rtol=1e-11,atol=1e-12)
    if not res.success:
        raise RuntimeError(res.message)
    return res.y[0]


def make_figures(p: Oscillator,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    t=np.linspace(0,32,4001)
    tr,ss,total=components(t,p)
    numerical=numerical_solution(t,p)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(t,total,label='Exact total response',lw=2)
    ax.plot(t,tr,label='Natural transient',alpha=.8)
    ax.plot(t,ss,label='Driven steady-state',alpha=.8)
    ax.plot(t,numerical,'--',label='Independent IVP solver',alpha=.65)
    ax.set(xlabel='Time (s)',ylabel='Displacement (m)',title='Two motions inside one spring: transient + steady state')
    ax.grid(alpha=.25);ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(out_dir/'decomposition.png',dpi=175);plt.close(fig)
    ps=modes(p)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.axhline(0,color='gray',lw=1);ax.axvline(0,color='gray',lw=1)
    ax.scatter(ps[:2].real,ps[:2].imag,marker='x',s=100,label='Natural system poles')
    ax.scatter(ps[2:].real,ps[2:].imag,marker='o',facecolors='none',edgecolors='tab:orange',s=100,label='Input poles')
    ax.set(xlabel='Real(s): decay / growth',ylabel='Imag(s): angular frequency',title='Natural poles versus forcing poles')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out_dir/'pole_map.png',dpi=175);plt.close(fig)
    w=np.linspace(0.01,4.5,1200)
    amps=p.force/np.sqrt((p.k-p.m*w*w)**2+(p.c*w)**2)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(w,amps,label='Steady-state amplitude')
    ax.axvline(p.omega,ls='--',color='gray',label='Selected forcing frequency')
    ax.axvline(np.sqrt(p.k/p.m),ls=':',color='tab:orange',label='Undamped natural frequency')
    ax.set(xlabel='Forcing angular frequency (rad/s)',ylabel='Amplitude (m)',title='Frequency response and resonance')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out_dir/'resonance.png',dpi=175);plt.close(fig)
    vals=[]
    for c in [.2,.6,1.2]:
        pp=Oscillator(m=p.m,c=c,k=p.k,force=p.force,omega=p.omega,x0=p.x0,v0=p.v0)
        tr0,ss0,total0=components(t,pp)
        vals.append((c,total0))
    fig,ax=plt.subplots(figsize=(9,4.5))
    for c,y in vals:ax.plot(t,y,label=f'c = {c}')
    ax.set(xlabel='Time (s)',ylabel='Displacement (m)',title='How damping changes the time-domain response')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out_dir/'damping_comparison.png',dpi=175);plt.close(fig)
    return float(np.max(np.abs(total-numerical)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='figures')
    parser.add_argument('--x0',type=float,default=0.0)
    parser.add_argument('--v0',type=float,default=0.0)
    parser.add_argument('--damping',type=float,default=0.6)
    parser.add_argument('--frequency',type=float,default=1.2)
    args=parser.parse_args()
    p=Oscillator(c=args.damping,omega=args.frequency,x0=args.x0,v0=args.v0)
    error=make_figures(p,args.out)
    A,phi,alpha,wd,C,D=coefficients(p)
    report={'parameters':p.__dict__,'natural_poles':[[float(z.real),float(z.imag)] for z in modes(p)[:2]],'forcing_poles':[[float(z.real),float(z.imag)] for z in modes(p)[2:]],'steady_state_amplitude':float(A),'phase_lag_radians':float(phi),'decay_envelope_rate':float(alpha),'damped_frequency':float(wd),'transient_C':float(C),'transient_D':float(D),'numerical_max_error':error,'one_percent_decay_envelope_time_seconds':float(np.log(100)/alpha) if alpha>0 else None}
    print(json.dumps(report,indent=2))
    Path(args.out,'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
