"""Production-scale persistence adapter; requires pip install ripser."""
import numpy as np
from .topology import normalize_rows

def rips_diagrams(weight_rows, normalize=True, maxdim=1):
    """For 64/128 point neuron clouds. Returns Ripser diagrams with essential H0.

    Use finite_interval() to exclude the single infinite H0 bar for summaries.
    """
    try:
        from ripser import ripser
    except ImportError as e:
        raise RuntimeError('Install optional dependency: python -m pip install ripser') from e
    x=np.asarray(weight_rows,dtype=float)
    if normalize: x=normalize_rows(x)
    return ripser(x, maxdim=maxdim)['dgms']

def finite_intervals(diagram):
    d=np.asarray(diagram,dtype=float).reshape((-1,2))
    return d[np.isfinite(d).all(axis=1)]
