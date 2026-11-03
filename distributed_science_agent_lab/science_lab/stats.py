"""Statistical workhorse and audit-friendly multiple-testing correction."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import ttest_ind

@dataclass(frozen=True)
class Finding:
    name: str
    feature_index: int
    discovery_p: float
    holdout_p: float
    effect_d: float
    holdout_effect_d: float
    discovery_n: int
    holdout_n: int
    provenance: str

    def to_dict(self):
        return asdict(self)


def effect_size(x0, x1):
    a, b = np.asarray(x0), np.asarray(x1)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def contrast(x, y, col: int):
    a, b = x[y == 0, col], x[y == 1, col]
    p = float(ttest_ind(a, b, equal_var=False).pvalue)
    return (p if np.isfinite(p) else 1.0), effect_size(a, b)


def inspect_feature(study, feature_index: int):
    p, d = contrast(study.x_train, study.y_train, feature_index)
    p_h, d_h = contrast(study.x_holdout, study.y_holdout, feature_index)
    return Finding(study.names[feature_index], feature_index, p, p_h, d, d_h,
                   len(study.y_train), len(study.y_holdout),
                   f'sklearn:load_wine:classes_0_1:seed_{study.seed}:feature_{feature_index}')


def benjamini_hochberg(pvalues):
    """Return monotonic BH adjusted p-values in original order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('p-values must be a finite 1-D vector in [0,1]')
    m = len(p)
    if m == 0:
        return p.copy()
    ix = np.argsort(p, kind='stable')
    ordered = np.minimum(1.0, p[ix] * m / np.arange(1, m+1))
    ordered = np.minimum.accumulate(ordered[::-1])[::-1]
    out = np.empty_like(p)
    out[ix] = ordered
    return out
