"""Open scikit-learn Wine measurements plus explicitly synthetic null controls."""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class Study:
    x_train: np.ndarray
    x_holdout: np.ndarray
    y_train: np.ndarray
    y_holdout: np.ndarray
    names: tuple[str, ...]
    observed_count: int
    seed: int


def make_study(seed: int = 0, null_count: int = 24, holdout_fraction: float = 0.3) -> Study:
    if null_count < 0 or not (0.1 <= holdout_fraction < 0.6):
        raise ValueError('Invalid study parameters')
    raw = load_wine()
    keep = raw.target != 2  # Class 0 versus 1, not a medical endpoint
    x = np.asarray(raw.data[keep], dtype=np.float64)
    y = np.asarray(raw.target[keep], dtype=np.int64)
    # Synthetic nulls have no intended association with the cultivar label.
    rng = np.random.default_rng(90000 + seed)
    x_null = rng.normal(size=(len(x), null_count))
    complete = np.concatenate([x, x_null], axis=1)
    # Select before analysis. No holdout observations are seen by the discoverer.
    tr, te = train_test_split(np.arange(len(y)), test_size=holdout_fraction,
                              stratify=y, random_state=seed)
    names = tuple(raw.feature_names) + tuple(f'synthetic_null_{i:02d}' for i in range(null_count))
    return Study(complete[tr], complete[te], y[tr], y[te], names, len(raw.feature_names), seed)
