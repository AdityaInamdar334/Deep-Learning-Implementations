from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Recording:
    spikes: tuple[np.ndarray, ...]
    duration_ms: float
    metadata: dict

    def __post_init__(self):
        if self.duration_ms <= 0:
            raise ValueError('duration must be positive')
        if not self.spikes:
            raise ValueError('at least one neuron required')
        for s in self.spikes:
            if s.ndim != 1 or not np.all(np.isfinite(s)):
                raise ValueError('spike times must be finite 1D arrays')
            if np.any(s < 0) or np.any(s >= self.duration_ms):
                raise ValueError('spike times outside recording')
            if len(s) > 1 and np.any(np.diff(s) <= 0):
                raise ValueError('spike times must be strictly increasing')

    @property
    def n_units(self):
        return len(self.spikes)

    @property
    def total_spikes(self):
        return sum(len(x) for x in self.spikes)

    def digest(self):
        h = hashlib.sha256()
        h.update(np.float64(self.duration_ms).tobytes())
        for x in self.spikes:
            h.update(np.int64(len(x)).tobytes())
            h.update(np.asarray(x, dtype='<f8').tobytes())
        h.update(json.dumps(self.metadata, sort_keys=True).encode())
        return h.hexdigest()

    def window(self, start_ms: float, end_ms: float):
        if not (0 <= start_ms < end_ms <= self.duration_ms):
            raise ValueError('invalid window')
        # Shift relative to window start, exclude right boundary.
        spikes = tuple((s[(s >= start_ms) & (s < end_ms)] - start_ms).copy() for s in self.spikes)
        return Recording(spikes, float(end_ms - start_ms),
                         {'parent_hash': self.digest(), 'window_ms': [start_ms, end_ms]})


def make_synthetic_recording(seed: int = 0, n_units: int = 12, duration_ms: float = 30000.0) -> Recording:
    """Synthetic Poisson spike trains + shared stimulus events. No biological measurements."""
    if n_units < 4 or duration_ms < 4000:
        raise ValueError('need >=4 units and >=4s duration')
    rng = np.random.default_rng(seed)
    cue_start, cue_end = duration_ms/3, 2*duration_ms/3
    spikes = []
    # Shared stimulus-associated event train for first four neurons.
    shared = np.arange(cue_start+250, cue_end-250, 360.0)
    for unit in range(n_units):
        base_rate = 3.5 + 0.8 * (unit % 5)
        n_background = rng.poisson(base_rate * duration_ms / 1000)
        all_spikes = rng.uniform(0, duration_ms, n_background)
        if unit < 4:
            extra = shared + rng.normal(0, 5.0, len(shared))
            all_spikes = np.concatenate([all_spikes, extra])
        # Some units increase their rates during cue; signal is NOT proof of causation.
        if unit % 4 == 0:
            n_cue = rng.poisson(4.5 * (cue_end-cue_start) / 1000)
            all_spikes = np.concatenate([all_spikes, rng.uniform(cue_start, cue_end, n_cue)])
        s = np.sort(all_spikes[(all_spikes >= 0) & (all_spikes < duration_ms)])
        spikes.append(np.unique(s).astype(np.float64))
    metadata = {'kind': 'fully_synthetic', 'seed': seed,
                'cue_ms': [cue_start, cue_end], 'time_unit': 'ms',
                'shared_event_units': [0,1,2,3]}
    return Recording(tuple(spikes), float(duration_ms), metadata)
