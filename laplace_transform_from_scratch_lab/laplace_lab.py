"""Laplace transform visual lab. Run: python laplace_lab.py"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)


def finite_exponential_integral(s, a=0j, T=25.):
    """Integral_0^T exp((a-s)t) dt, computed stably close to s=a."""
    z=complex(a)-complex(s)
    if abs(z) < 1e-12: return complex(T)
    return np.expm1(z*T)/z


def exponential_laplace(s, a=0j):
    """Analytic expression, equals actual unilateral integral only if Re(s)>Re(a)."""
    return 1/(complex(s)-complex(a))


def cosine_laplace(s, omega=1.):
    """Analytic form; unilateral integral converges for Re(s)>0."""
    return s/(s*s+omega*omega)


def oscillator(m=1., c=.6, k=4., omega=1.2):
    """Zero initial conditions: x''m+c x'+k x = cos(omega*t)"""
    denominator=np.polymul([m,c,k], [1,0,omega*omega])
    numerator=np.array([1.,0.])
    r,p,d=signal.residue(numerator,denominator)
    return r,p,d, denominator


def oscillator_response(t, m=1., c=.6, k=4., omega=1.2):
    t=np.asarray(t,float)
    r,p,_,denom=oscillator(m,c,k,omega)
    via_poles=np.real(np.sum(r[:,None]*np.exp(p[:,None]*t[None,:]),axis=0))
    system=signal.TransferFunction([1.],[m,c,k])
    via_ode_time,via_ode,_=signal.lsim(system,U=np.cos(omega*t),T=t)
    return via_poles,via_ode,p,r


def plot_spirals():
    vals=[.6+.3j,.15+1j,.7-1.2j]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    t=np.linspace(0,35,3000)
    for ax,s in zip(axes,vals):
        z=(1-np.exp(-s*t))/s
        ax.plot(z.real,z.imag,lw=1.65)
        ax.scatter([0,z[-1].real],[0,z[-1].imag],s=[18,36])
        ax.set(title=f's = {s}',xlabel='real integral',ylabel='imaginary integral')
        ax.axis('equal');ax.grid(alpha=.25)
    fig.suptitle('The complex integral accumulates as a curve')
    fig.tight_layout();fig.savefig(FIG/'integral_paths.png',dpi=170);plt.close(fig)


def plot_poles():
    sigma=np.linspace(-1,1,380)
    omega=np.linspace(-3,3,440)
    S=sigma[None,:]+1j*omega[:,None]
    F=S/(S*S+1)
    mag=np.clip(np.abs(F),0,12)
    fig,ax=plt.subplots(figsize=(8,5.4))
    im=ax.pcolormesh(sigma,omega,np.log10(1+mag),shading='auto',cmap='viridis')
    ax.scatter([0,0],[1,-1],marker='x',s=130,color='red',linewidth=2,label='poles ±i')
    ax.axvline(0,color='white',linestyle='--',linewidth=1,alpha=.8)
    ax.set(xlabel='Re(s)',ylabel='Im(s)',title='Analytic continuation of L{cos(t)}: poles at ±i')
    ax.legend(loc='upper right');fig.colorbar(im,ax=ax,label='log10(1 + capped magnitude)')
    fig.tight_layout();fig.savefig(FIG/'cosine_poles.png',dpi=170);plt.close(fig)


def plot_oscillator():
    t=np.linspace(0,32,3201)
    solution,numeric,poles,residues=oscillator_response(t)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(t,solution,label='sum of four exponential modes',linewidth=2)
    ax.plot(t,numeric,'--',label='numerical ODE solution',alpha=.8)
    ax.plot(t,np.cos(1.2*t),':',label='driving force cos(1.2t)',alpha=.45)
    ax.set(xlabel='time',ylabel='displacement',title='Forced damped oscillator: poles reconstruct motion')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIG/'damped_oscillator.png',dpi=170);plt.close(fig)
    print('Oscillator poles:',np.round(poles,5))
    print('Residues:',np.round(residues,5))
    print('Max difference from numerical ODE:',float(np.max(np.abs(solution-numeric))))


def plot_convergence():
    a=0
    T=np.linspace(.01,28,420)
    ss=[1+1j, .1+.3j, -.1+.3j]
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    for s in ss:
        result=np.array([finite_exponential_integral(s,a,x) for x in T])
        ax.plot(T,np.abs(result),label=f'Re(s)={s.real:+.1f}, Im(s)={s.imag:+.1f}')
    ax.set(xlabel='upper integration limit T',ylabel='|integral 0..T|',title='Convergence depends on Re(s) > Re(a)')
    ax.set_ylim(0,20);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIG/'region_of_convergence.png',dpi=170);plt.close(fig)


def main():
    plot_spirals();plot_poles();plot_oscillator();plot_convergence()
    print('Figures saved to',FIG)
    print('At s=1+i, transform of 1:',exponential_laplace(1+1j))
    print('Finite integral with T=25:',finite_exponential_integral(1+1j,T=25))

if __name__=='__main__': main()
