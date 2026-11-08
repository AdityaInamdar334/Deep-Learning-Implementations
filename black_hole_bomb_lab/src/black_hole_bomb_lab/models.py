"""Explicitly illustrative reduced models; published numbers are annotated below."""
from __future__ import annotations
import numpy as np

# Švančara et al., arXiv:2609.05595v1, Fig. 1e and main text.
GAMMA_PER_S = 0.081       # measured 0.081 +/- 0.001 s^-1
PLATEAU_M = 0.00306      # measured radially-averaged 3.06 +/- 0.05 mm
FREQ_HZ = 3.47           # measured low-amplitude sloshing-mode frequency, Hz
FREQ_DRIFT_HZ = 0.098   # reported drift magnitude, Hz (direction not inferred)
KAPPA_M2_S = 9.98e-8    # helium-4 circulation quantum as cited in the paper
C_M2_S = 19.2e-4        # paper's fitted vortex C, Eq. 1 convention
RADIUS_M = 0.0373
DEPTH_M = 0.0216
DRAIN_DIAMETER_M = 0.010
A0_M = 20e-6             # CHOSEN toy seed, not an experimental measurement


def linear_amplitude(t, initial=A0_M, gamma=GAMMA_PER_S):
    """Linear small-signal envelope exp(gamma*t), no nonlinear limitation."""
    return np.asarray(initial) * np.exp(gamma * np.asarray(t, dtype=float))


def saturated_amplitude(t, initial=A0_M, gamma=GAMMA_PER_S,
                        plateau=PLATEAU_M):
    """Analytic amplitude of illustrative Stuart–Landau normal form.

    dA/dt=gamma*A-beta*A^3, beta=gamma/plateau^2.
    This equation is NOT a dynamical law extracted or fitted by the experiment.
    """
    t=np.asarray(t, dtype=float)
    if not 0 < initial < plateau:
        raise ValueError('initial must satisfy 0 < initial < plateau')
    if gamma<=0 or plateau<=0: raise ValueError('gamma and plateau must be positive')
    ratio = (plateau / initial) ** 2 - 1.0
    return plateau / np.sqrt(1.0 + ratio*np.exp(-2.0*gamma*t))


def mode_amplitudes(t, initial=A0_M, harmonic_fraction=0.20, static_fraction=0.07):
    """Illustrative higher harmonic scales A^2; static response also A^2.

    Fractions are arbitrary demonstration settings; NOT the paper's mode fits.
    """
    a=saturated_amplitude(t, initial=initial)
    return a, harmonic_fraction*a**2/PLATEAU_M, static_fraction*a**2/PLATEAU_M


def circulation_quanta(c=C_M2_S, kappa=KAPPA_M2_S):
    """Number of circulation quanta for v_theta=Omega*r + C/r."""
    return 2 * np.pi * c / kappa


def coupled_energy(t, initial=A0_M, gamma=GAMMA_PER_S,
                   plateau=PLATEAU_M, secondary_decay=0.65):
    """Minimal *hypothesis* model: pumping -> primary -> secondary -> sink.

    P=A^2, S=secondary proxy energy (m^2); dP/dt=2γP-λP²,
    dS/dt=λP²-δS, with λ=2γ/plateau².
    The transfer term is an illustrative closure, NOT a measured energy budget.
    Integrate the secondary ODE by trapezoidal convolution / time stepping.
    """
    t=np.asarray(t,dtype=float)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or t[0]<0:
        raise ValueError('t must be 1D, increasing, nonnegative, >=2 samples')
    if secondary_decay<=0: raise ValueError('secondary_decay must be positive')
    a=saturated_amplitude(t,initial=initial,gamma=gamma,plateau=plateau)
    p=a*a
    rate=2*gamma/plateau**2
    transfer=rate*p*p
    s=np.zeros_like(t)
    # exact substep for a linearly interpolated source (trapezoidal small-dt approximation)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        damping=np.exp(-secondary_decay*dt)
        s[i]=damping*s[i-1]+0.5*dt*(damping*transfer[i-1]+transfer[i])
    injected=2*gamma*p
    dissipation=secondary_decay*s
    return {'time_s':t,'primary':p,'secondary':s,'injection':injected,
            'transfer':transfer,'dissipation':dissipation}


def bicoherence_experiment(n_segments=128, fs=128, seconds=2.0, f0=3.5, seed=11):
    """Toy phase-locking example: bicoherence of f+f=2f, not paper's g3.

    Independent segment phases reduce coherent bispectral averaging.
    All numbers are SYNTHETIC. 3.5 Hz chosen to land on FFT bins.
    """
    rng=np.random.default_rng(seed)
    n=int(fs*seconds)
    t=np.arange(n)/fs
    bin1=round(f0*seconds)
    if abs(bin1/seconds-f0)>1e-10 or bin1*2>=n//2:
        raise ValueError('Choose f0 aligned to an FFT bin below Nyquist/2')
    coherent=[]; incoherent=[]
    for _ in range(n_segments):
        phi=rng.uniform(0,2*np.pi)
        base=np.cos(2*np.pi*f0*t+phi)
        locked=base+.35*np.cos(4*np.pi*f0*t+2*phi+0.3)
        unrelated=base+.35*np.cos(4*np.pi*f0*t+rng.uniform(0,2*np.pi))
        coherent.append(np.fft.rfft(locked)[[bin1,2*bin1]])
        incoherent.append(np.fft.rfft(unrelated)[[bin1,2*bin1]])
    def bicoh(a):
        a=np.asarray(a)
        p=a[:,0]*a[:,0]
        q=a[:,1]
        return float(np.abs(np.mean(p*np.conj(q))) / np.sqrt(np.mean(np.abs(p)**2)*np.mean(np.abs(q)**2)))
    return {'coherent':bicoh(coherent),'random_phase':bicoh(incoherent),
            'segments':n_segments,'frequency_hz':f0}
