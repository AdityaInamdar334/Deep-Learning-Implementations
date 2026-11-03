"""Auditable educational spike-statistics methods; not a replacement for official SpikeLab."""
import numpy as np
from .data import Recording


def _time_covered(spikes: np.ndarray, duration_ms: float, delta_ms: float) -> float:
    """Fraction of [0,T] within +-delta of any spike, merging overlapping intervals."""
    if len(spikes) == 0:
        return 0.0
    lo = np.maximum(spikes - delta_ms, 0.0)
    hi = np.minimum(spikes + delta_ms, duration_ms)
    start, stop, total = float(lo[0]), float(hi[0]), 0.0
    for a, b in zip(lo[1:], hi[1:]):
        if a > stop:
            total += stop - start
            start, stop = float(a), float(b)
        else:
            stop = max(stop, float(b))
    return (total + stop - start) / duration_ms


def _coincident_fraction(source: np.ndarray, other: np.ndarray, delta_ms: float) -> float:
    if len(source) == 0 or len(other) == 0:
        return float('nan')
    idx = np.searchsorted(other, source)
    left = np.clip(idx-1, 0, len(other)-1)
    right = np.clip(idx, 0, len(other)-1)
    close = np.minimum(np.abs(source - other[left]), np.abs(source - other[right]))
    return float(np.mean(close <= delta_ms))


def sttc(a: np.ndarray, b: np.ndarray, duration_ms: float, delta_ms: float = 20.0) -> float:
    """Spike Time Tiling Coefficient (Cutts & Eglen), finite nonempty spike trains.

    Returns NaN for an empty train, an undefined edge case to report, not silently zero.
    """
    if duration_ms <= 0 or not (0 < delta_ms < duration_ms/2):
        raise ValueError('invalid duration or coincidence window')
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a)==0 or len(b)==0:
        return float('nan')
    for s in (a,b):
        if (s[0] < 0 or s[-1] >= duration_ms or np.any(np.diff(s) <= 0)):
            raise ValueError('sorted spike times in [0,duration) required')
    pa = _coincident_fraction(a,b,delta_ms)
    pb = _coincident_fraction(b,a,delta_ms)
    ta = _time_covered(a,duration_ms,delta_ms)
    tb = _time_covered(b,duration_ms,delta_ms)
    def term(p,t):
        den = 1.0 - p*t
        return (p-t)/den if abs(den)>1e-12 else 1.0
    return float(np.clip(0.5*(term(pa,tb)+term(pb,ta)), -1, 1))


def bin_spikes(recording: Recording, bin_ms: float = 50.0) -> np.ndarray:
    if bin_ms <= 0 or recording.duration_ms/bin_ms < 2:
        raise ValueError('invalid bin size')
    edges = np.arange(0, recording.duration_ms, bin_ms)
    edges = np.append(edges, recording.duration_ms) if edges[-1] < recording.duration_ms else edges
    if len(edges) < 3:
        raise ValueError('insufficient bins')
    return np.array([np.histogram(s, bins=edges)[0] for s in recording.spikes], dtype=float)


def population_coupling(recording: Recording, bin_ms: float = 50.0) -> np.ndarray:
    """Pearson correlation of each binned unit with population excluding that unit.

    This is an educational rate-correlation proxy, NOT the full event-triggered
    coupling estimator used in SpikeLab's paper. Zero-variance -> NaN.
    """
    counts = bin_spikes(recording, bin_ms)
    out = np.empty(recording.n_units, dtype=float)
    total = counts.sum(axis=0)
    for i in range(recording.n_units):
        x, y = counts[i], total - counts[i]
        out[i] = float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else float('nan')
    return out


def circular_shift_surrogates(a, b, duration_ms, delta_ms=20.0, n_surrogates=39, seed=0):
    """Shift one unit by random offsets, wrap to recording, preserve count.

    Conditioning on units with stationary dynamics is required for interpretation.
    """
    if n_surrogates < 1: raise ValueError('n_surrogates >= 1')
    observed = sttc(a,b,duration_ms,delta_ms)
    if not np.isfinite(observed):
        return {'observed': None, 'p_two_sided': None, 'null_mean': None}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_surrogates):
        shift = float(rng.uniform(2*delta_ms, duration_ms-2*delta_ms))
        rotated = np.sort((np.asarray(b)+shift) % duration_ms)
        null.append(sttc(a,rotated,duration_ms,delta_ms))
    null=np.asarray(null)
    p=(1+int(np.sum(np.abs(null)>=abs(observed))))/(n_surrogates+1)
    return {'observed': float(observed), 'p_two_sided': float(p), 'null_mean': float(np.mean(null))}


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted q values, retaining NaNs as unknown."""
    p=np.asarray(pvalues,float)
    q=np.full(p.shape, np.nan)
    finite=np.flatnonzero(np.isfinite(p))
    if len(finite)==0: return q
    vals=p[finite]
    if np.any((vals<0)|(vals>1)): raise ValueError('p values out of range')
    order=np.argsort(vals)
    raw=vals[order]*len(vals)/np.arange(1,len(vals)+1)
    adjusted=np.minimum.accumulate(raw[::-1])[::-1]
    q[finite[order]]=np.minimum(adjusted,1.)
    return q


def pca_variance(recording: Recording, bin_ms: float = 50.0, max_components: int = 5):
    """Population rate PCA via SVD; not an HMM or state-space model."""
    x=bin_spikes(recording,bin_ms).T
    x=x-x.mean(axis=0)
    _, s, _ = np.linalg.svd(x,full_matrices=False)
    power=s**2
    if power.sum()==0: return [float('nan')]
    return (power[:max_components]/power.sum()).tolist()
